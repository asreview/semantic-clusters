#!/usr/bin/python
# -*- coding: utf-8 -*-
# Path: asreviewcontrib\semantic_clustering\main.py

# Environment imports
import sys
import getopt

# Local Imports
from semantic_clustering import SemanticClustering
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
