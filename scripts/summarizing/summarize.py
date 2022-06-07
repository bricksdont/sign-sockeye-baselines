#! /usr/bin/python3

import os
import argparse
import logging
import itertools
import operator

from typing import List, Tuple


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--eval-folder", type=str, help="Path that should be searched for results.",
                        required=True)

    args = parser.parse_args()

    return args


def walklevel(some_dir, level=1):
    """
    Taken from:
    https://stackoverflow.com/a/234329/1987598
    :param some_dir:
    :param level:
    :return:
    """
    some_dir = some_dir.rstrip(os.path.sep)

    assert os.path.isdir(some_dir)
    num_sep = some_dir.count(os.path.sep)
    for root, dirs, files in os.walk(some_dir):
        yield root, dirs, files
        num_sep_this = root.count(os.path.sep)
        if num_sep + level <= num_sep_this:
            del dirs[:]


def parse_filename(filename: str) -> Tuple[str, str]:
    """
    Structure:  $corpus.target.$metric
    Example:    unseen.target.bleu

    :param filename:
    :return:
    """
    parts = filename.split(".")

    if len(parts) != 3:
        logging.error("Cannot parse filename: '%s'" % filename)

    corpus, target_string, metric = parts

    assert target_string == "trg", "Cannot parse filename: '%s'" % filename

    assert corpus in ["test", "unseen"], "Cannot parse filename: '%s'" % filename

    return corpus, metric


def read_bleu(filename: str) -> str:
    """

    :param filename:
    :return:
    """
    with open(filename, "r") as infile:
        line = infile.readline().strip()

        parts = line.split(" ")

    if len(parts) < 3:
        return "-"

    return parts[2]


def read_chrf(filename: str) -> str:
    """
    Example content: #chrF2+numchars.6+space.false+version.1.4.14 = 0.47
    :param filename:
    :return:
    """

    with open(filename, "r") as infile:
        line = infile.readline().strip()

        parts = line.split(" ")

    if len(parts) < 3:
        return "-"

    return parts[2]


def read_metric_values(metric: str, filepath: str):
    """

    :param metric:
    :param filepath:
    :return:
    """
    if metric == "bleu":
        metric_names = ["BLEU"]
        metric_values = [read_bleu(filepath)]
    elif metric == "chrf":
        metric_names = ["CHRF"]
        metric_values = [read_chrf(filepath)]
    else:
        raise NotImplementedError

    return metric_names, metric_values


def parse_model_name(model_name: str) -> Tuple[str, str, str, str, str]:
    """
    Examples:

    training_corpus.focusnews+force_target_fps.true+normalize_poses.true+pose_type.mediapipe
    dry_run

    :param model_name:
    :return:
    """
    training_corpus, force_target_fps, normalize_poses, pose_type, bucket_scaling = "-", "-", "-", "-", "-"

    if "dry_run" in model_name:
        return training_corpus, force_target_fps, normalize_poses, pose_type, bucket_scaling

    pairs = model_name.split("+")

    for pair in pairs:
        key, value = pair.split(".")

        if key == "training_corpus":
            training_corpus = value
        elif key == "force_target_fps":
            force_target_fps = value
        elif key == "normalize_poses":
            normalize_poses = value
        elif key == "pose_type":
            pose_type = value
        elif key == "bucket_scaling":
            bucket_scaling = value
        else:
            logging.warning("Could not parse (key, value:): %s, %s", key, value)
            raise NotImplementedError

    return training_corpus, force_target_fps, normalize_poses, pose_type, bucket_scaling


class Result(object):

    def __init__(self,
                 langpair,
                 model_name,
                 corpus,
                 training_corpus,
                 test_src,
                 test_trg,
                 force_target_fps,
                 normalize_poses,
                 pose_type,
                 bucket_scaling,
                 metric_names,
                 metric_values):
        self.langpair = langpair
        self.model_name = model_name
        self.corpus = corpus
        self.training_corpus = training_corpus
        self.test_src = test_src
        self.test_trg = test_trg
        self.force_target_fps = force_target_fps
        self.normalize_poses = normalize_poses
        self.pose_type = pose_type
        self.bucket_scaling = bucket_scaling
        self.metric_dict = {}

        self.update_metrics(metric_names, metric_values)

    def update_metrics(self,
                       metric_names,
                       metric_values):
        for name, value in zip(metric_names, metric_values):
            self.update_metric(name, value)

    def update_metric(self, metric_name, metric_value):
        assert metric_name not in self.metric_dict.keys(), "Refusing to overwrite existing metric key!"
        self.metric_dict[metric_name] = metric_value

    def __repr__(self):
        metric_dict = str(self.metric_dict)

        return "Result(%s)" % "+".join([self.langpair,
                                        self.model_name,
                                        self.corpus,
                                        self.training_corpus,
                                        self.test_src,
                                        self.test_trg,
                                        self.force_target_fps,
                                        self.normalize_poses,
                                        self.pose_type,
                                        self.bucket_scaling,
                                        metric_dict])

    def signature(self) -> str:
        return "+".join([self.langpair,
                         self.model_name,
                         self.corpus,
                         self.training_corpus,
                         self.test_src,
                         self.test_trg,
                         self.force_target_fps,
                         self.normalize_poses,
                         self.pose_type,
                         self.bucket_scaling])


def collapse_metrics(results: List[Result]) -> Result:
    """
    :param results:
    :return:
    """
    first_result = results[0]

    for r in results[1:]:
        for name, value in r.metric_dict.items():
            first_result.update_metric(name, value)

    return first_result


def reduce_results(results: List[Result]) -> List[Result]:
    """
    :param results:
    :return:
    """

    with_signatures = [(r.signature(), r) for r in results]  # type: List[Tuple[str, Result]]
    with_signatures.sort(key=operator.itemgetter(0))

    by_signature_iterator = itertools.groupby(with_signatures, operator.itemgetter(0))

    reduced_results = []

    for signature, subiterator in by_signature_iterator:
        subresults = [r for s, r in subiterator]
        reduced_result = collapse_metrics(subresults)
        reduced_results.append(reduced_result)

    return reduced_results


def get_subdirectories(eval_folder: str) -> List[str]:
    """

    :param eval_folder:
    :return:
    """

    langpairs = []

    for filename in os.listdir(eval_folder):
        filepath = os.path.join(eval_folder, filename)
        if os.path.isdir(filepath):
            langpairs.append(filename)

    return langpairs


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    results = []

    langpairs = get_subdirectories(args.eval_folder)

    logging.debug("Language pairs:")
    logging.debug(langpairs)

    for langpair_index, langpair in enumerate(langpairs):

        test_src, test_trg = langpair.split("-")

        path_langpair = os.path.join(args.eval_folder, langpair)

        model_names = get_subdirectories(path_langpair)

        if langpair_index == 0:
            logging.debug("Model names:")
            logging.debug(model_names)

        for model_name in model_names:
            path_model = os.path.join(path_langpair, model_name)

            training_corpus, force_target_fps, normalize_poses, pose_type, bucket_scaling = parse_model_name(model_name)

            for _, _, files in os.walk(path_model):
                for file in files:
                    corpus, metric = parse_filename(file)

                    filepath = os.path.join(path_model, file)

                    metric_names, metric_values = read_metric_values(metric, filepath)

                    result = Result(langpair,
                                    model_name,
                                    corpus,
                                    training_corpus,
                                    test_src,
                                    test_trg,
                                    force_target_fps,
                                    normalize_poses,
                                    pose_type,
                                    bucket_scaling,
                                    metric_names,
                                    metric_values)

                    results.append(result)
                    logging.debug("Found result: %s", result)

    results = reduce_results(results)

    header_names = ["LANGPAIR",
                    "MODEL_NAME",
                    "CORPUS",
                    "TRAINING_CORPUS",
                    "TEST_SRC",
                    "TEST_TRG",
                    "FORCE_TARGET_FPS",
                    "NORMALIZE_POSES",
                    "POSE_TYPE",
                    "BUCKET_SCALING",
                    "BLEU",
                    "CHRF"]

    metric_names = ["BLEU",
                    "CHRF"]

    print("\t".join(header_names))

    for r in results:
        values = [r.langpair, r.model_name, r.corpus, r.training_corpus, r.test_src, r.test_trg,
                  r.force_target_fps, r.normalize_poses, r.pose_type, r.bucket_scaling]
        metrics = [r.metric_dict.get(m, "-") for m in metric_names]

        print("\t".join(values + metrics))


if __name__ == '__main__':
    main()
