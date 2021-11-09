#!/usr/bin/python

# Copyright 2021 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from semantic_clustering import SemanticClustering
import sys
import getopt
from interactive import run_app
from asreview.data import ASReviewData


def main(argv):
    filepath = ""

    try:
        opts, args = getopt.getopt(
            argv, "htf:a", ["help", "testfile", "filepath=", "app"])
    except getopt.GetoptError:
        print('Please use the following format:')
        print('test.py -f <filepath>')
        print('test.py --testfile')
        print('test.py --app')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('test.py -f <filepath> or --testfile')
            sys.exit()
        elif opt in ("-f", "--filepath"):
            filepath = arg
        elif opt in ("-t", "--testfile"):
            filepath = "https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv"
        elif opt in ("-a", "--app"):
            run_app()
            sys.exit(1)
    print('Running from file: ', filepath)

    # check if arguments are empty
    if filepath == "":
        print('Please use the following format:')
        print('test.py -f <filepath>')
        print('test.py --testfile')
        print('test.py --app')
        sys.exit(2)

    SemanticClustering(ASReviewData.from_file(filepath))


if __name__ == "__main__":
    main(sys.argv[1:])
