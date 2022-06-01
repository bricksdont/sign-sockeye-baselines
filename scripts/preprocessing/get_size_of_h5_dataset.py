#! /usr/bin/python3

import sys

from sockeye import h5_io


assert len(sys.argv) > 1

filename = sys.argv[1]

reader = h5_io.H5Reader(filename=filename)

num_examples = len(list(reader.iterate()))

print(num_examples)
