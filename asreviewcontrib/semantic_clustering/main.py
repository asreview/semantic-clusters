#!/usr/bin/python
# -*- coding: utf-8 -*-
# Path: asreviewcontrib\semantic_clustering\main.py

import argparse
import sys
import os

import webbrowser

from asreview.data import ASReviewData
from asreview.entry_points import BaseEntryPoint
from asreviewcontrib.semantic_clustering.interactive import run_app
from asreviewcontrib.semantic_clustering.semantic_clustering import SemanticClustering  # noqa: E501


class SemClusEntryPoint(BaseEntryPoint):
    description = "Semantic clustering tools for ASReview."
    extension_name = "asreview-semantic-clustering"

    def __init__(self):
        self.version = "0.1"

    def execute(self, argv):
        args = _parse_arguments(
            version=f"{self.extension_name}: {self.version}", argv=argv)

        if args.filepath:
            data = ASReviewData.from_file(args.filepath.name)
            SemanticClustering(
                data,
                args.output,
                transformer=args.transformer)

        elif args.testfile:
            data = ASReviewData.from_file("https://raw.githubusercontent.com/asreview/systematic-review-datasets/master/datasets/van_de_Schoot_2017/output/van_de_Schoot_2017.csv")  # noqa: E501
            SemanticClustering(
                data,
                args.output,
                transformer=args.transformer)

        elif args.app:
            url = "http://127.0.0.1:8050/"
            webbrowser.open(url, new=2, autoraise=True)
            run_app(args.app)
        sys.exit(1)


# check file extension
def _valid_file(param):
    base, ext = os.path.splitext(param)
    if ext.lower() not in ('.csv'):
        raise argparse.ArgumentTypeError('File must have a csv extension')
    return param


# argument parser
def _parse_arguments(version="Unknown", argv=None):
    parser = argparse.ArgumentParser(prog='asreview semantic clustering')
    group = parser.add_mutually_exclusive_group()

    group.add_argument(
        "-f",
        "--filepath",
        metavar="INPUT FILEPATH",
        help="processes the specified file",
        type=argparse.FileType('r', encoding='UTF-8')
    )
    group.add_argument(
        "-t",
        "--testfile",
        help="uses a test file instead of providing a file",
        action="store_true",
    )
    group.add_argument(
        "-a",
        "--app",
        metavar="INPUT FILEPATH",
        help="runs the app from a file created with the semantic clustering "
        "extension",
        type=argparse.FileType('r', encoding='UTF-8')
    )

    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version="%(prog)s " + version,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="output file name",
        metavar="OUTPUT FILE NAME",
        type=str,
        default="output.csv"
    )

    parser.add_argument(
        "--transformer",
        help="select a transformer to use",
        metavar="TRANSFORMER",
        type=str,
        default='allenai/scibert_scivocab_uncased'
    )

    # Exit if no arguments are given
    if len(argv) == 0:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args(argv)

    # Check if the file extension is correct
    if args.app is not None:
        _valid_file(args.app.name)

    if args.output is not None:
        _valid_file(args.output)

    return args
