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
from espnet.utils.cli_utils import strtobool

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
    parser.add_argument('--dtype', choices=("float16", "float32", "float64"), default="float32",
                        help='Float precision (only available in --api v2)')
    parser.add_argument('--backend', type=str, default='pytorch',
                        choices=['pytorch'],
                        help='Backend library')
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
    parser.add_argument('--api', default="v1", choices=["v1", "v2"],
                        help='''Beam search APIs
        v1: Default API. It only supports the ASRInterface.recognize method and DefaultRNNLM.
        v2: Experimental API. It supports any models that implements ScorerInterface.''')

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
                        choices=[1, 2],
                        help='Number of speakers in the speech')
    parser.add_argument('--num-encs', default=1, type=int,
                        help='Number of encoders in the model.')
    parser.add_argument('--encoder-type', type=str, default="transformer",
                        choices=["transformer", "rnn"],
                        help='encoder type is transformer for lstm based')

    # search related
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')
    parser.add_argument('--nbest-copy-input', action='store_true', default=False,
                        help='Copy input to output json')
    parser.add_argument('--nbest-compute-wer', action='store_true', default=False,
                        help='Calculate wer for partial hypothesis of the nbest list')
    parser.add_argument('--beam-size', type=int, default=1,
                        help='Beam size')
    parser.add_argument('--penalty', type=float, default=0.0,
                        help='Insertion penalty (Weiran: actually bonus)')
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
                        CTC/attention decoding especially on GPU. Smaller margin
                        makes decoding faster, but may increase search errors.
                        If margin=0 (default), this function is disabled""")
    # context for input (low frame rate)
    parser.add_argument('--input-context', default=0, type=int,
                        help='Context to use at input.')
    parser.add_argument('--input-skiprate', default=1, type=int,
                        help='Skiprate to use at input.')
    # transducer related
    parser.add_argument('--score-norm-transducer', type=strtobool, nargs='?',
                        default=True,
                        help='Normalize transducer scores by length')

    # rnnlm related
    parser.add_argument('--rnnlm', type=str, default=None,
                        help='RNNLM model file to read')
    parser.add_argument('--rnnlm-conf', type=str, default=None,
                        help='RNNLM model config file to read')
    parser.add_argument('--lexicon-dict', type=str, default=None,
                        help='Lexicon dict to read')

    parser.add_argument('--word-rnnlm', type=str, default=None,
                        help='Word RNNLM model file to read')
    parser.add_argument('--word-rnnlm-conf', type=str, default=None,
                        help='Word RNNLM model config file to read')
    parser.add_argument('--word-dict', type=str, default=None,
                        help='Word list to read')

    parser.add_argument('--lm-weight', type=float, default=0.1,
                        help='RNNLM weight')
    parser.add_argument('--word-bonus', type=float, default=0.0,
                        help='word bonus')
    parser.add_argument('--truth-file', type=str, default=None,
                        help='file containing ground truth text')

    # streaming related
    parser.add_argument('--streaming-scoring-ratio', type=int, default=5,
                        help="Multiple of beamsize for CTC prefix search during streaming decoding")
    parser.add_argument('--streaming-ctc-blank-offset', type=float, default=0.0,
                        help='Offset for posterior to likelihood transition')
    parser.add_argument('--streaming-ctc-maxactive', type=int, default=100,
                        help='Max active states for CTC prefix search')
    parser.add_argument('--streaming-beam-width', type=float, default=50.0,
                        help="Beam width for aggressive search during streaming decoding")
    parser.add_argument('--streaming-att-delay', type=int, default=6,
                        help='Estimate of span of attention per token')
    parser.add_argument('--streaming-dec-delay', type=int, default=0,
                        help='Number of frames for look-ahead by decoder')

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

    # display PYTHONPATH
    logging.info('python path = ' + os.environ.get('PYTHONPATH', '(None)'))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    logging.info('set random seed = %d' % args.seed)

    # validate rnn options
    if args.word_rnnlm is not None:
        raise NotImplementedError("Streaming decoder does not support word-rnnlm yet")

    logging.info('backend = ' + args.backend)
    if args.num_spkrs == 1:
        if not args.num_encs == 1:
            raise NotImplementedError("Streaming decoding only supports single encoder")
        if args.dtype != "float32":
            raise NotImplementedError(f"`--dtype {args.dtype}` is only available with `--api v2`")

        from espnet.asr.pytorch_backend.decode_streaming import recog
        recog(args)
    else:
        raise NotImplementedError("Streaming decoding only supports single speaker")

if __name__ == '__main__':
    main(sys.argv[1:])
