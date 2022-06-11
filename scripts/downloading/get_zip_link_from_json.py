#! /usr/bin/python3

import argparse
import logging
import json


"""
Structure:

{
   "conceptdoi":"10.5281/zenodo.6621479",
   "conceptrecid":"6621479",
   "created":"2022-06-09T17:04:20.104254+00:00",
   "doi":"10.5281/zenodo.6621480",
   "files":[
      {
         "bucket":"65c30721-1e0d-4094-9f51-ba70acd6dca4",
         "check
sum":"md5:095e15edd6bd39950727c6c2692ed727",
         "key":"focusnews.zip",
         "links":{
            "self":"https://zenodo.org/api/files/65c30721-1e0d-4094-9f51-ba70acd6dca4/focusnews.zip"
         },
         "size":18502247314,
         "type":"zip"
      }
   ],
   "id":6621480,
   "links":{ ... },
   "metadata":{ ... }
}
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input", type=str, help="Path to JSON response.", required=True)

    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    logging.basicConfig(level=logging.DEBUG)
    logging.debug(args)

    with open(args.input) as infile:
        json_dict = json.load(infile)

    file_dict = json_dict["files"][0]

    assert file_dict["type"] == "zip"

    link = file_dict["links"]["self"]

    print(link)


if __name__ == '__main__':
    main()
