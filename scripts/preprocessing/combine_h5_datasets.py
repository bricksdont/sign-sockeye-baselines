#! /usr/bin/python3

import argparse
import logging


# noinspection PyUnresolvedReferences
from sockeye import h5_io


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--inputs", type=str, nargs="+",
                        help="Paths to 2 or more h5 datasets.", required=True)
    parser.add_argument("--output", type=str,
                        help="Path where combined dataset should be stored.", required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    readers = [h5_io.H5Reader(filename=input_path) for input_path in args.inputs]

    writer = h5_io.H5Writer(filename=args.output)

    for reader in readers:
        for sample in reader.iterate():
            writer.add(sample)

    writer.close()

    for reader in readers:
        reader.close()


if __name__ == '__main__':
    main()
