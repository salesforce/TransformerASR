"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import argparse
parser=argparse.ArgumentParser()
parser.add_argument("--infile", default="", help="input file name", type=str)
parser.add_argument("--outfile", default="", help="output file name", type=str)
parser.add_argument("--phonefile", default="", help="phone file name", type=str)
parser.add_argument("--mode", default="p2c", help="conversion mode, p2c (phone-to-char) or c2p (char-to-phone)", type=str)
parser.add_argument("--in_separator", default="", help="separator for separating input", type=str)
parser.add_argument("--out_separator", default="", help="separator for joining output", type=str)

args=parser.parse_args()

import string
ALPHABET=list(string.ascii_letters + string.digits)

def readFile(fname):
    with open(fname) as fin:
        retval  = fin.read().rstrip('\n').split("\n")
    return retval

if __name__ == "__main__":

    # The first argument is the phone set.
    list_phones = readFile(args.phonefile)
    num_phones = len(list_phones)
    assert num_phones <= len(ALPHABET), "Not enough alphabet to convert all phones"
    
    # Luckily, we do not have more than 62 phones in the phone set.
    list_chars = ALPHABET[:num_phones]
    dict_phone_to_char = dict(zip(list_phones, list_chars))
    dict_char_to_phone = dict(zip(list_chars, list_phones))
    with open("/".join(args.phonefile.split("/")[:-1]) + "/map_phone_to_char", "w") as fout:
        results = [("%s %s" % (k, dict_phone_to_char[k])) for k in dict_phone_to_char]
        fout.write("\n".join(results))

    with open("/".join(args.phonefile.split("/")[:-1]) + "/map_char_to_phone", "w") as fout:
        results = [("%s %s" % (k, dict_char_to_phone[k])) for k in dict_char_to_phone]
        fout.write("\n".join(results))

    if args.mode == "p2c":
        dict_phone_to_char[args.in_separator] = args.out_separator

        fout = open(args.outfile, "w")
        for line in readFile(args.infile):
            fields = line.split()
            utt = fields[0]
            val = [dict_phone_to_char[x] for x in fields[1:]]
            fout.write("%s %s\n" % (utt, "".join(val)))
        fout.close()
        
    else:
        if args.mode == "c2p":
            dict_char_to_phone[args.in_separator] = args.out_separator

            fout = open(args.outfile, "w")
            for line in readFile(args.infile):
                fields = line.split()
                utt = fields[0]
                val = [dict_char_to_phone[x] for x in fields[1:]]
                fout.write("%s %s\n" % (utt, "".join(val)))
            fout.close()


