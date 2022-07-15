#! /usr/bin/python3

import os
import cv2
import srt
import argparse
import datetime
import logging

from typing import Tuple


def get_num_frames_and_fps(filename: str) -> Tuple[int, int]:
    """
    https://stackoverflow.com/a/60976166/1987598
    """
    cap = cv2.VideoCapture(filename)
    fps = cap.get(cv2.CAP_PROP_FPS)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    return int(num_frames), fps


def get_duration(filename: str) -> float:
    """
    https://stackoverflow.com/a/60976166/1987598
    """
    num_frames, fps = get_num_frames_and_fps(filename)
    duration = float(num_frames) / float(fps)

    return duration


def get_subtitle_name(video_name: str) -> str:

    parts = video_name.split(".")

    without_extension = ".".join(parts[:-1])

    return without_extension + ".srt"


def write_dummy_subtitle(filepath: str, duration: float):

    start_time = datetime.timedelta(seconds=0.0)
    end_time = datetime.timedelta(seconds=duration)

    subtitle = srt.Subtitle(content="Dummy string",
                            start=start_time,
                            end=end_time,
                            index=1)

    with open(filepath, "w") as handle:
        handle.write(srt.compose([subtitle]))


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--download-sub", type=str,
                        help="Input folder with original download data which has subfolders"
                             " 'openpose', 'mediapipe' and 'videos' - but not 'subtitles'.", required=True)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    video_folder = os.path.join(args.download_sub, "videos")
    subtitle_folder = os.path.join(args.download_sub, "subtitles")

    os.makedirs(subtitle_folder, exist_ok=True)

    for video_name in os.listdir(video_folder):

        video_path = os.path.join(video_folder, video_name)

        duration = get_duration(video_path)

        subtitle_name = get_subtitle_name(video_name)
        subtitle_path = os.path.join(subtitle_folder, subtitle_name)

        write_dummy_subtitle(filepath=subtitle_path, duration=duration)


if __name__ == '__main__':
    main()
