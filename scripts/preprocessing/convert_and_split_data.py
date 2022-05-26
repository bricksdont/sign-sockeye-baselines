#! /usr/bin/python3

import os
import datetime
import srt
import tarfile
import tempfile
import argparse
import logging

import numpy as np

from typing import List, Dict, Iterator, Tuple, Optional

# noinspection PyUnresolvedReferences
from sockeye import h5_io
from pose_format import Pose
from pose_format.utils.openpose import load_openpose_directory


def extract_tar_xz_file(filename: str, target_dir: str):
    """

    :param filename:
    :param target_dir:
    :return:
    """
    with tarfile.open(filename) as tar_handle:
        tar_handle.extractall(path=target_dir)


def read_openpose_surrey_format(filename: str, fps: int) -> Pose:
    """
    Read files of the form "focusnews.071.openpose.tar.xz"

    :param filename:
    :param fps:
    :return:
    """
    with tempfile.TemporaryDirectory(prefix="extract_pose_file") as tmpdir_name:
        # extract tar.xz
        extract_tar_xz_file(filename=filename, target_dir=tmpdir_name)

        # load directory
        poses = load_openpose_directory(directory=tmpdir_name, fps=fps)

    return poses


def get_file_id(filename: str) -> str:
    """
    Examples:
    - srf.2020-03-12.srt
    - focusnews.120.srt

    :param filename:
    :return:
    """
    parts = filename.split(".")

    return parts[1]


def read_subtitles(subtitle_dir: str) -> Dict[str, List[srt.Subtitle]]:
    """

    :param subtitle_dir:
    :return:
    """

    subtitles_by_id = {}  # type: Dict[str, List[srt.Subtitle]]

    for filename in os.listdir(subtitle_dir):
        filepath = os.path.join(subtitle_dir, filename)

        file_id = get_file_id(filename)

        subtitles = []  # type: List[srt.Subtitle]

        with open(filepath, "r") as handle:
            for subtitle in srt.parse(handle.read()):
                subtitles.append(subtitle)

        subtitles_by_id[file_id] = subtitles

    return subtitles_by_id


def miliseconds_to_frame_index(miliseconds: int, fps: int) -> int:
    """
    :param miliseconds:
    :param fps:
    :return:
    """
    return int(fps * (miliseconds / 1000))


def convert_srt_time_to_frame(srt_time: datetime.timedelta, fps: int) -> int:
    """
    datetime.timedelta(seconds=4, microseconds=71000)

    :param srt_time:
    :param fps:
    :return:
    """
    seconds, microseconds = srt_time.seconds, srt_time.microseconds

    miliseconds = int((seconds * 1000) + (microseconds / 1000))

    return miliseconds_to_frame_index(miliseconds=miliseconds, fps=fps)


def reduce_pose_slice(pose_slice: np.array) -> np.array:
    """
    Keep only the first person and reduce to 2 dimensions

    :param pose_slice:
    :return:
    """

    # alternative: brackets around [0] to keep dimension
    pose_slice = pose_slice[:, 0, :, :]

    # collapse to 1 vector for each frame
    return pose_slice.reshape(pose_slice.shape[0], -1)


def extract_parallel_examples(subtitles: List[srt.Subtitle],
                              poses: Pose,
                              fps: int) -> Iterator[Tuple[str, np.array]]:
    """

    :param subtitles: Example:
                      Subtitle(index=1,
                               start=datetime.timedelta(seconds=33, microseconds=843000),
                               end=datetime.timedelta(seconds=38, microseconds=97000),
                               content='地球上只有3%的水是淡水', proprietary='')
    :param poses: Array dimensions: (frames, person, points, dimensions)
    :param fps:
    :return:
    """

    for subtitle in subtitles:
        start_frame = convert_srt_time_to_frame(subtitle.start, fps=fps)
        end_frame = convert_srt_time_to_frame(subtitle.end, fps=fps)

        pose_slice = poses.body.data[start_frame:end_frame]
        pose_slice = reduce_pose_slice(pose_slice)

        yield subtitle.content, pose_slice


class ParallelWriter:

    def __init__(self, data_sub: str, pose_type: str, prefix: str, max_size: Optional[int] = None):
        """

        :param data_sub:
        :param pose_type:
        :param prefix:
        :param max_size:
        """
        assert prefix in ["train", "dev", "test"]

        self.prefix = prefix
        self.data_sub = data_sub
        self.pose_type = pose_type
        self.max_size = max_size

        text_output_name = ".".join([self.prefix, "txt"])
        self.text_output_path = os.path.join(self.data_sub, text_output_name)
        self.text_writer = open(self.text_output_path, "w")

        poses_output_name = ".".join([self.prefix, self.pose_type, "h5"])
        self.poses_output_path = os.path.join(self.data_sub, poses_output_name)
        self.pose_writer = sockeye.h5_io.H5Writer(filename=self.poses_output_path)

        self.size = 0

    def close(self):
        self.text_writer.close()
        self.pose_writer.close()

    def add(self, text: str, pose_slice: np.array):

        if self.max_size is not None:
            assert self.size < self.max_size, "Reached maximum size of %d, refusing to add more examples." % self.max_size

        self.text_writer.write(text)
        self.pose_writer.add(pose_slice)

        self.size += 1

    @property
    def writable(self) -> bool:

        if self.max_size is None:
            return True

        return self.max_size > self.size


def decide_on_split(num_examples: int,
                    train_size: int,
                    devtest_size: int,
                    writers: Dict[str, ParallelWriter]) -> Dict[int, ParallelWriter]:
    """

    :param num_examples:
    :param train_size:
    :param devtest_size:
    :param writers:
    :return:
    """

    train_indexes = np.arange(num_examples, dtype=np.int32)

    # sub-sample if train_size has a limit

    if train_size != -1:
        train_indexes = np.random.choice(train_indexes, size=(train_size,), replace=False)

    # default: training writer

    writers_by_id = {index: writers["train"] for index in train_indexes}

    # sample indexes for dev

    dev_indexes = np.random.choice(train_indexes, size=(devtest_size,), replace=False)

    for dev_index in dev_indexes:
        writers_by_id[dev_index] = writers["dev"]

    # sample indexes for test

    remaining_train_indexes = np.asarray([i for i in train_indexes if i not in dev_indexes])

    test_indexes = np.random.choice(remaining_train_indexes, size=(devtest_size,), replace=False)

    for test_index in test_indexes:
        writers_by_id[test_index] = writers["test"]

    return writers_by_id


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--download-sub", type=str,
                        help="Input folder with original download data which has subfolders 'subtitles'"
                             " 'openpose' and 'mediapipe'.", required=True)
    parser.add_argument("--data-sub", type=str,
                        help="Output folder to store converted and split data sets.", required=True)

    parser.add_argument("--seed", type=int,
                        help="Random seed for data splits.", required=True)
    parser.add_argument("--train-size", type=int, default=None,
                        help="Maximum number of examples in train set. Default: no limit.", required=False)
    parser.add_argument("--devtest-size", type=int,
                        help="Number of examples in dev and test set each.", required=True)

    parser.add_argument("--fps", type=int,
                        help="Framerate.", required=True)
    parser.add_argument("--pose-type", type=str,
                        help="Type of poses (openpose or mediapipe).", required=True, choices=["openpose", "mediapipe"])

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    np.random.seed(args.seed)

    writer_train = ParallelWriter(data_sub=args.data_sub,
                                  pose_type=args.pose_type,
                                  prefix="train",
                                  max_size=args.train_size)

    writer_dev = ParallelWriter(data_sub=args.data_sub,
                                pose_type=args.pose_type,
                                prefix="dev",
                                max_size=args.devtest_size)

    writer_test = ParallelWriter(data_sub=args.data_sub,
                                 pose_type=args.pose_type,
                                 prefix="test",
                                 max_size=args.devtest_size)

    writers = {"train": writer_train, "dev": writer_dev, "test": writer_test}

    # load all subtitles (since they don't use a lot of memory)

    subtitle_dir = os.path.join(args.download_sub, "subtitles")
    subtitles_by_id = read_subtitles(subtitle_dir)

    num_examples = sum([len(subtitles) for subtitles in subtitles_by_id.values()])
    writers_by_id = decide_on_split(num_examples=num_examples,
                                    train_size=args.train_size,
                                    devtest_size=args.devtest_size,
                                    writers=writers)

    # step through poses one by one
    pose_dir = os.path.join(args.download_sub, args.pose_type)

    filename: str

    example_id = 0

    for filename in os.listdir(pose_dir):
        file_id = get_file_id(filename)

        if "openpose" in filename:
            poses = read_openpose_surrey_format(filename=filename, fps=args.fps)
        else:
            raise NotImplementedError

        matching_subtitles = subtitles_by_id[file_id]

        for text, pose_slice in extract_parallel_examples(poses=poses, subtitles=matching_subtitles, fps=args.fps):

            writer = writers_by_id[example_id]
            writer.add(text=text, pose_slice=pose_slice)

            example_id += 1

    for writer in writers.values():
        writer.close()


if __name__ == '__main__':
    main()
