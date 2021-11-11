#!/usr/bin/python
# -*- coding: utf-8 -*-
# Path: asreviewcontrib\semantic_clustering\main.py

# Environment imports
import sys
import getopt

# Local imports
from asreviewcontrib.semantic_clustering.semantic_clustering import SemanticClustering
from asreviewcontrib.semantic_clustering.interactive import run_app

# ASReview imports
from asreview.data import ASReviewData
from asreview.entry_points import BaseEntryPoint


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
            filepath = "https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv"  # noqa: E501
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


class SemClusEntryPoint(BaseEntryPoint):
    description = "Semantic clustering tools for ASReview."
    extension_name = "asreview-semantic-clustering"

    def __init__(self, args):
        super().__init__()
        self.version = "0.1"

    def execute(self, argv):
        main(argv)
