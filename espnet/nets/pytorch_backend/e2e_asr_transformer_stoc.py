"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# Copyright 2019 Shigeki Karita
# Copyright 2020 Weiran Wang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

from argparse import Namespace
from distutils.util import strtobool

import logging

import math
import numpy as np
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import kron
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_stoc import Decoder
from espnet.nets.pytorch_backend.transformer.encoder_stoc import Encoder
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import LabelSmoothingLoss
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.mask import make_context_mask
from espnet.nets.scorers.ctc import CTCPrefixScorer
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.transformer.subsampling import _context_concat

CTC_LOSS_THRESHOLD = 10000
CTC_SCORING_RATIO = 1.5

class E2E(ASRInterface, torch.nn.Module):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        group = parser.add_argument_group("transformer model setting")

        group.add_argument("--transformer-init", type=str, default="pytorch",
                           choices=["pytorch", "xavier_uniform", "xavier_normal",
                                    "kaiming_uniform", "kaiming_normal"],
                           help='how to initialize transformer parameters')
        group.add_argument("--transformer-input-layer", type=str, default="conv2d",
                           choices=["conv2d", "linear", "embed", "conv2d_with_deltas", "conv2d_1layer_with_deltas", "conv2d_yingbo"],
                           help='transformer input layer type')
        group.add_argument('--transformer-attn-dropout-rate', default=None, type=float,
                           help='dropout in transformer attention. use --dropout-rate if None is set')
        group.add_argument('--transformer-length-normalized-loss', default=True, type=strtobool,
                           help='normalize loss by length')

        # Weiran: for the noam optimizer.
        group.add_argument('--transformer-lr', default=10.0, type=float,
                           help='Initial value of learning rate')
        group.add_argument('--transformer-warmup-steps', default=25000, type=int,
                           help='optimizer warmup steps')

        # Weiran: for the rampup optimizer.
        group.add_argument('--rampup-sr', default=1000, type=int,
                           help='steps to rampup')
        group.add_argument('--rampup-si', default=40000, type=int,
                           help='steps to hold')
        group.add_argument('--rampup-sf', default=160000, type=int,
                           help='steps to decay')
        group.add_argument('--rampup-lr', default=0.001, type=float,
                           help='peak learning rate for the rampup optimizer')

        # Weiran: added death rates for stochastic layers.
        group.add_argument('--edeath-rate', default=0.1, type=float,
                           help='death rate for encoder')
        group.add_argument('--ddeath-rate', default=0.1, type=float,
                           help='death rate for decoder')

        # Encoder
        group.add_argument('--dropout-rate', default=0.0, type=float,
                           help='Dropout rate for the encoder')
        group.add_argument('--elayers', default=4, type=int,
                           help='Number of encoder layers (for shared recognition part in multi-speaker asr mode)')
        group.add_argument('--eunits', '-u', default=300, type=int,
                           help='Number of encoder hidden units')

        # Attention
        group.add_argument('--adim', default=320, type=int,
                           help='Number of attention transformation dimensions')
        group.add_argument('--aheads', default=4, type=int,
                           help='Number of heads for multi head attention')
        # Decoder
        group.add_argument('--dlayers', default=1, type=int,
                           help='Number of decoder layers')
        group.add_argument('--dunits', default=320, type=int,
                           help='Number of decoder hidden units')

        # Trigger CTC for training the streaming model.
        group.add_argument('--streaming_encoder', default=False, type=strtobool,
                           help='Whether to use streaming transformer encoder')
        group.add_argument('--streaming_left_context', default=0, type=int,
                           help='How many previous frames to attend to')
        group.add_argument('--streaming_right_context', default=0, type=int,
                           help='How many future frames to attend to')
        group.add_argument('--streaming_block_size', default=0, type=int,
                           help='How many frames are given to encoder at once')

        group.add_argument('--streaming_dec_context', default=0, type=int,
                           help='How many previous tokens to attend to')

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        self.mtlalpha = args.mtlalpha

        self.subsample = [1]
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            death_rate=args.edeath_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate
        )
        self.enc_odim = args.adim

        # Weiran: streaming setup.
        if hasattr(args, 'streaming_encoder'):
            self.streaming_encoder = args.streaming_encoder
            self.streaming_left_context = args.streaming_left_context
            self.streaming_right_context = args.streaming_right_context
        else:
            self.streaming_encoder = False
            self.streaming_left_context = 0
            self.streaming_right_context = 0

        if hasattr(args, 'streaming_block_size'):
            self.streaming_block_size = args.streaming_block_size
        else:
            self.streaming_block_size = 0

        if hasattr(args, 'streaming_dec_context'):
            self.streaming_dec_context = args.streaming_dec_context
        else:
            self.streaming_dec_context = 0

        if self.mtlalpha < 1.0:
            self.decoder = Decoder(
                odim=odim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                death_rate=args.ddeath_rate,
                positional_dropout_rate=args.dropout_rate,
                self_attention_dropout_rate=args.transformer_attn_dropout_rate,
                src_attention_dropout_rate=args.transformer_attn_dropout_rate
            )
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.reporter = None

        self.criterion = LabelSmoothingLoss(self.odim, self.ignore_id, args.lsm_weight,
                                            args.transformer_length_normalized_loss)
        self.reset_parameters(args)
        self.adim = args.adim

        if args.mtlalpha > 0.0:
            self.ctc = CTC(odim, self.enc_odim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        else:
            self.ctc = None

        # errors for attn model are computed based on teacher-forcing.
        if args.report_cer:
            self.error_calculator = ErrorCalculator(args.char_list,
                                                    args.sym_space, args.sym_blank,
                                                    args.report_cer, report_wer=False)
        else:
            self.error_calculator = None
        self.rnnlm = None

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel

        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        if self.streaming_encoder:
            # src_mask has shape (B, 1, maxlen), and context_mask has shape (1, maxlen, maxlen)
            context_mask = make_context_mask(src_mask.size(-1), self.streaming_left_context, self.streaming_right_context, device=src_mask.device).unsqueeze(0)
            if self.streaming_block_size > 0:
                input_maxlen = max(ilens)
                num_blocks = int(math.ceil(float(input_maxlen) / self.streaming_block_size))
                block_mask = kron(torch.eye(num_blocks), torch.ones([self.streaming_block_size, self.streaming_block_size]))[:input_maxlen, :input_maxlen]
                context_mask = context_mask | (block_mask.unsqueeze(0).bool().to(context_mask.device))
            src_mask = src_mask & context_mask.to(src_mask.device)

        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad
        batch_size = hs_pad.size(0)
        max_len_h = hs_pad.size(1)
        # pick only the first slice of time_in to be consistent as before, as we do not need the full form of mask later.
        if self.streaming_encoder:
            didx = torch.arange(batch_size).unsqueeze(1) * (max_len_h**2) + torch.arange(0, max_len_h**2, max_len_h+1).unsqueeze(0)
            hs_mask = torch.take(hs_mask, didx.to(hs_mask.device).view(-1)).view([batch_size, 1, max_len_h])

        hs_len = hs_mask.view(batch_size, max_len_h).sum(1)

        # 2. forward decoder
        if self.mtlalpha == 1.0:
            loss_att = 0.0
            self.acc = 0.0
        else:
            ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
            ys_mask = target_mask(ys_in_pad, self.ignore_id)
            if self.streaming_dec_context > 0:
                context_mask = make_context_mask(ys_mask.size(-1), self.streaming_dec_context, 0, device=ys_mask.device).unsqueeze(0)
                ys_mask = ys_mask & context_mask
            pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, hs_pad, hs_mask)
            self.pred_pad = pred_pad

            # 3. compute attention loss
            loss_att = self.criterion(pred_pad, ys_out_pad)
            self.acc = th_accuracy(pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id)

        # TODO(karita) show predicted text
        # TODO(karita) calculate these stats
        if self.mtlalpha == 0.0:
            loss_ctc = 0.0
        else:
            batch_size = xs_pad.size(0)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.enc_odim), hs_len, ys_pad)

        # 5. compute cer/wer
        if self.training or (self.error_calculator is None) or (self.mtlalpha == 0.0):
            cer_ctc = 0.0
        else:
            ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.enc_odim)).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)

        # copied from e2e_asr
        alpha = self.mtlalpha
        if alpha == 0.0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(0.0)
        elif alpha == 1.0:
            self.loss = loss_ctc
            loss_att_data = float(0.0)
            loss_ctc_data = float(loss_ctc)
        else:
            self.loss = alpha * loss_ctc + (1 - alpha) * loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            return self.loss, loss_data, loss_ctc_data, loss_att_data, float(self.acc), float(cer_ctc)
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
            return self.loss, loss_data, loss_ctc_data, loss_att_data, float(self.acc), float(cer_ctc)

    def scorers(self):
        """Scorers."""
        return dict(decoder=self.decoder, ctc=CTCPrefixScorer(self.ctc, self.eos))

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        if self.streaming_encoder:
            context_mask = make_context_mask(x.size(1), self.streaming_left_context, self.streaming_right_context, device=x.device).unsqueeze(0)
            if self.streaming_block_size > 0:
                input_maxlen = x.size(1)
                num_blocks = int(math.ceil(float(input_maxlen) / self.streaming_block_size))
                block_mask = kron(torch.eye(num_blocks), torch.ones([self.streaming_block_size, self.streaming_block_size]))[:input_maxlen, :input_maxlen]
                context_mask = context_mask | (block_mask.unsqueeze(0).bool().to(context_mask.device))
        else:
            context_mask = None
        enc_output, _ = self.encoder(x, context_mask)
        return enc_output.squeeze(0)

    def recognize(self, x, recog_args, char_list=None, rnnlm=None, use_jit=False):
        """Recognize input speech.
        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """

        enc_output = self.encode(x).unsqueeze(0)
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(enc_output)
            lpz = lpz.squeeze(0)
        else:
            lpz = None

        h = enc_output.squeeze(0)
        T = h.shape[0]

        logging.info('input lengths: ' + str(h.size(0)))
        # search parms
        beam = recog_args.beam_size
        penalty = recog_args.penalty
        ctc_weight = recog_args.ctc_weight

        # prepare sos
        y = self.sos
        vy = h.new_zeros(1).long()

        if recog_args.maxlenratio == 0:
            maxlen = h.shape[0]
        else:
            maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
        minlen = int(recog_args.minlenratio * h.size(0))
        logging.info('max output length: ' + str(maxlen))
        logging.info('min output length: ' + str(minlen))

        # initialize hypothesis
        if rnnlm:
            hyp = {'score': 0.0, 'yseq': [y], 'rnnlm_prev': None}
        else:
            hyp = {'score': 0.0, 'yseq': [y]}
        if lpz is not None:
            from espnet.nets.ctc_prefix_score import CTCPrefixScore

            ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, self.eos, np)
            hyp['ctc_state_prev'] = ctc_prefix_score.initial_state()
            hyp['ctc_score_prev'] = 0.0
            if ctc_weight != 1.0:
                # pre-pruning based on attention scores
                ctc_beam = min(lpz.shape[-1], int(beam * CTC_SCORING_RATIO))
            else:
                ctc_beam = lpz.shape[-1]
        hyps = [hyp]
        ended_hyps = []

        import six
        traced_decoder = None
        for i in six.moves.range(maxlen):
            logging.debug('position ' + str(i))

            hyps_best_kept = []
            for hyp in hyps:
                vy[0] = hyp['yseq'][i]

                # get nbest local scores and their ids
                ys_mask = subsequent_mask(i + 1).unsqueeze(0)
                ys = torch.tensor(hyp['yseq']).unsqueeze(0)
                # FIXME: jit does not match non-jit result
                if use_jit:
                    if traced_decoder is None:
                        traced_decoder = torch.jit.trace(self.decoder.forward_one_step,
                                                         (ys, ys_mask, enc_output))
                    local_att_scores = traced_decoder(ys, ys_mask, enc_output)[0]
                else:
                    local_att_scores = self.decoder.forward_one_step(ys, ys_mask, enc_output)[0]

                if rnnlm:
                    rnnlm_state, local_lm_scores = rnnlm.predict(hyp['rnnlm_prev'], vy)
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                if lpz is not None:
                    local_best_scores, local_best_ids = torch.topk(local_att_scores, ctc_beam, dim=1)
                    ctc_scores, ctc_states = ctc_prefix_score(hyp['yseq'], local_best_ids[0], hyp['ctc_state_prev'])

                    local_scores = \
                        (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                        + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc_score_prev'])
                    if rnnlm:
                        local_scores += recog_args.lm_weight * local_lm_scores[:, local_best_ids[0]]
                    local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                    local_best_ids = local_best_ids[:, joint_best_ids[0]]
                else:
                    local_best_scores, local_best_ids = torch.topk(local_scores, beam, dim=1)

                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state
                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(
                    hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

            # sort and get nbest
            hyps = hyps_best_kept
            logging.debug('number of pruned hypothes: ' + str(len(hyps)))
            if char_list is not None:
                logging.debug(
                    'best hypo: ' + ''.join([char_list[int(x)] for x in hyps[0]['yseq'][1:]]))

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                logging.info('adding <eos> in the last postion in the loop')
                for hyp in hyps:
                    if not hyp['yseq'][-1] == self.eos:
                        hyp['yseq'].append(self.eos)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a problem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == self.eos:
                    # only store the sequence that has more than minlen outputs
                    # also add penalty
                    if len(hyp['yseq']) > minlen:
                        hyp['score'] += (i + 1) * penalty
                        if rnnlm:  # Word LM needs to add final <eos> score
                            hyp['score'] += recog_args.lm_weight * rnnlm.final(hyp['rnnlm_prev'])
                        ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            # end detection
            from espnet.nets.e2e_asr_common import end_detect
            if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
                logging.info('end detected at %d', i)
                break

            hyps = remained_hyps
            if len(hyps) > 0:
                logging.debug('remained hypothes: ' + str(len(hyps)))
            else:
                logging.info('no hypothesis. Finish decoding.')
                break

            if char_list is not None:
                for hyp in hyps:
                    logging.debug(
                        'hypo: ' + ''.join([char_list[int(x)] for x in hyp['yseq'][1:]]))

            logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

        nbest_hyps = sorted(
            ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

        # check number of hypothesis
        if len(nbest_hyps) == 0:
            logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
            # should copy because Namespace will be overwritten globally
            recog_args = Namespace(**vars(recog_args))
            recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
            return self.recognize(x, recog_args, char_list, rnnlm)

        logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
        logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))

        return nbest_hyps

    def calculate_all_attentions(self, xs_pad, ilens, ys_pad):
        """E2E attention calculation.

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor ys_pad: batch of padded token id sequence tensor (B, Lmax)
        :return: attention weights with the following shape,
            1) multi-head case => attention weights (B, H, Lmax, Tmax),
            2) other case => attention weights (B, Lmax, Tmax).
        :rtype: float ndarray
        """
        with torch.no_grad():
            self.forward(xs_pad, ilens, ys_pad)
        ret = dict()
        for name, m in self.named_modules():
            if isinstance(m, MultiHeadedAttention):
                ret[name] = m.attn.cpu().numpy()
        return ret
