#!/usr/bin/env python3
# encoding: utf-8

"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Copyright 2020 Salesforce Research (Weiran Wang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""End-to-end speech recognition model decoding script."""

import configargparse
import logging
import os
import random
import sys

import numpy as np

def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description='Transcribe text from speech using a speech recognition model on one CPU or GPU',
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
    # general configuration
    parser.add('--config', is_config_file=True,
               help='Config file path')
    parser.add('--config2', is_config_file=True,
               help='Second config file path that overwrites the settings in `--config`')
    parser.add('--config3', is_config_file=True,
               help='Third config file path that overwrites the settings in `--config` and `--config2`')

    parser.add_argument('--ngpu', type=int, default=0,
                        help='Number of GPUs')
    parser.add_argument('--backend', type=str, default='pytorch',
                        choices=['pytorch'], help='Backend library')
    parser.add_argument('--debugmode', type=int, default=1,
                        help='Debugmode')
    parser.add_argument('--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('--verbose', '-V', type=int, default=1,
                        help='Verbose option')
    parser.add_argument('--batchsize', type=int, default=1,
                        help='Batch size for beam search (0: means no batch processing)')
    parser.add_argument('--preprocess-conf', type=str, default=None,
                        help='The configuration file for the pre-processing')

    # task and dataloader related
    parser.add_argument('--recog-sets', type=str, default=None,
                        help='Recognition datasets')
    parser.add_argument('--precomputed-feats-type', type=str, default=None,
                        help='Audio feature type (e.g., fbank_pitch)')
    parser.add_argument('--spmodel', type=str, default=None,
                        help='Filename of the sentencepiece model for converting text to tokens')
    parser.add_argument('--text-filename', type=str, default=None,
                        help='File containing normalized text for training (to be processed by spmodel)')
    parser.add_argument('--loader-num-worker', type=int, default=0,
                        help='Number of workers for each data loader')
    parser.add_argument('--loader-cache-mb', type=int, default=2048,
                        help='Cache size (in mb) used by each loader worker')
    parser.add_argument('--num_replicas', type=int, default=1,
                        help='Total number of CPU jobs for parallel decoding')
    parser.add_argument('--jobid', type=int, default=1,
                        help='The jobid for current CPU decoding')
    parser.add_argument('--result-label', type=str, required=True,
                        help='Filename of result label data (json)')

    # model (parameter) related
    parser.add_argument('--model', type=str, required=True,
                        help='Model file parameters to read')
    parser.add_argument('--model-conf', type=str, default=None,
                        help='Model config file')
    parser.add_argument('--num-spkrs', type=int, default=1,
                        choices=[1],
                        help='Number of speakers in the speech')
    parser.add_argument('--num-encs', default=1, type=int,
                        help='Number of encoders in the model.')
    # search related
    parser.add_argument('--nbest', type=int, default=10,
                        help='Output N-best hypotheses')
    parser.add_argument('--nbest-copy-input', action='store_true', default=False,
                        help='Copy input to output json')
    parser.add_argument('--nbest-compute-wer', action='store_true', default=False,
                        help='Calculate wer for partial hypothesis of the nbest list')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty', type=float, default=0.0,
                        help='Incertion penalty')
    parser.add_argument('--maxlenratio', type=float, default=0.0,
                        help="""Input length ratio to obtain max output length.
                        If maxlenratio=0.0 (default), it uses a end-detect function
                        to automatically find maximum hypothesis lengths""")
    parser.add_argument('--minlenratio', type=float, default=0.0,
                        help='Input length ratio to obtain min output length')
    parser.add_argument('--ctc-weight', type=float, default=0.0,
                        help='CTC weight in joint decoding')
    parser.add_argument('--weights-ctc-dec', type=float, action='append',
                        help='ctc weight assigned to each encoder during decoding.[in multi-encoder mode only]')
    parser.add_argument('--ctc-window-margin', type=int, default=0,
                        help="""Use CTC window with margin parameter to accelerate
                        CTC/attention decoding especially on GPU. Smaller magin
                        makes decoding faster, but may increase search errors.
                        If margin=0 (default), this function is disabled""")

    # context for input (low frame rate)
    parser.add_argument('--input-context', default=0, type=int,
                        help='Context to use at input.')
    parser.add_argument('--input-skiprate', default=1, type=int,
                        help='Skiprate to use at input.')

    # rnnlm related
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--rnnlm-conf', type=str, default=None,
                        help='RNNLM model config file to read')
    parser.add_argument('--word-rnnlm', type=str, default=None,
                        help='Word RNNLM model file to read')
    parser.add_argument('--word-rnnlm-conf', type=str, default=None,
                        help='Word RNNLM model config file to read')
    parser.add_argument('--word-dict', type=str, default=None,
                        help='Word list to read')
    parser.add_argument('--lexicon-dict', type=str, default=None,
                        help='Lexicon dict to read')
    parser.add_argument('--lm-weight', type=float, default=0.1,
                        help='RNNLM weight')
    parser.add_argument('--sublm-weight', type=float, default=0.8,
                        help='Weight of subword LM in multi-level LM')
    parser.add_argument('--word-bonus', type=float, default=0.0,
                        help='Word bonus score')
    parser.add_argument('--truth_file', type=str, default=None,
                        help='file containing groud truth text')

    # realigner related
    parser.add_argument('--realigner-num-steps', default=-1, type=int,
                        help='Number of steps for iterative re-alignment')

    return parser


def main(args):
    """Run the main decoding function."""
    parser = get_parser()
    args = parser.parse_args(args)

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    elif args.verbose == 2:
        logging.basicConfig(level=logging.DEBUG,
                            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
    else:
        logging.basicConfig(
            level=logging.WARN, format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s")
        logging.warning("Skip DEBUG/INFO messages")

    # check CUDA_VISIBLE_DEVICES
    if args.ngpu > 0:
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cvd is None:
            logging.warning("CUDA_VISIBLE_DEVICES is not set.")
        elif args.ngpu != len(cvd.split(",")):
            logging.error("#gpus is not matched with CUDA_VISIBLE_DEVICES.")
            sys.exit(1)

        # TODO(mn5k): support of multiple GPUs
        if args.ngpu > 1:
            logging.error("The program only supports ngpu=1.")
            sys.exit(1)

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info('set random seed = %d' % args.seed)

    # validate rnn options
    if args.rnnlm is not None and args.word_rnnlm is not None:
        logging.warning("Both --rnnlm and --word-rnnlm are specified, using multi-level RNNLM.")

    # recog
    from espnet.asr.pytorch_backend.asr_dsl import recog
    recog(args)



if __name__ == '__main__':
    main(sys.argv[1:])
