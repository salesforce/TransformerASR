"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# Weiran:
# Warning! This script selects only one pronunciation for each word.

import sys

rootDir = "/export/home/voice-research/AcousticModeling"
sys.path.append("%s/util/" % rootDir)
from weirangadgets import readFile

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--infile1", default="", help="first json", type=str)
parser.add_argument("--infile2", default="", help="second json", type=str)
parser.add_argument("--infile3", default="", help="third json", type=str)
parser.add_argument("--infile4", default="", help="fourth json", type=str)
parser.add_argument("--infile5", default="", help="fifth json", type=str)
parser.add_argument("--infile6", default="", help="sixth json", type=str)
parser.add_argument("--outfile", default="", help="output file name", type=str)
args=parser.parse_args()

if __name__ == "__main__":


    new_lexicon = {}
    print("combining %s" % args.infile1)
    for line in readFile(args.infile1).split("\n"):
        fields = line.lower().split()
        if len(fields) > 1 and fields[0] not in new_lexicon:
            new_lexicon[fields[0]] = fields[1:]

    if args.infile2:
        print("combining %s" % args.infile2)
        for line in readFile(args.infile2).split("\n"):
            fields = line.lower().split()
            if len(fields) > 1 and fields[0] not in new_lexicon:
                new_lexicon[fields[0]] = fields[1:]

    if args.infile3:
        print("combining %s" % args.infile3)
        for line in readFile(args.infile3).split("\n"):
            fields = line.lower().split()
            if len(fields) > 1 and fields[0] not in new_lexicon:
                new_lexicon[fields[0]] = fields[1:]

    if args.infile4:
        print("combining %s" % args.infile4)
        for line in readFile(args.infile4).split("\n"):
            fields = line.lower().split()
            if len(fields) > 1 and fields[0] not in new_lexicon:
                new_lexicon[fields[0]] = fields[1:]

    if args.infile5:
        print("combining %s" % args.infile5)
        for line in readFile(args.infile5).split("\n"):
            fields = line.lower().split()
            if len(fields) > 1 and fields[0] not in new_lexicon:
                new_lexicon[fields[0]] = fields[1:]

    if args.infile6:
        print("combining %s" % args.infile6)
        for line in readFile(args.infile6).split("\n"):
            fields = line.lower().split()
            if len(fields) > 1 and fields[0] not in new_lexicon:
                new_lexicon[fields[0]] = fields[1:]

    with open(args.outfile, 'w') as fout:
        for w in new_lexicon:
            fout.write("%s %s\n" % (w, " ".join(new_lexicon[w])))

"""
This file is used in compile_extended_lexicon. 
"""