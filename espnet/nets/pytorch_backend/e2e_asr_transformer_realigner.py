"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

"""Transformer speech recognition model (pytorch)."""

import pdb
from argparse import Namespace
from distutils.util import strtobool
import torch.distributed as dist

import logging
logging.basicConfig(level=logging.DEBUG)

import math
import numpy as np
import torch

from espnet.nets.asr_interface import ASRInterface
from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.encoder_stoc import Encoder
from espnet.nets.pytorch_backend.transformer.decoder_stoc import Decoder
from espnet.nets.pytorch_backend.transformer.embedding import ScaledPositionalEncoding
from espnet.nets.pytorch_backend.transformer.initializer import initialize
from espnet.nets.pytorch_backend.transformer.subsampling import _context_concat
from espnet.nets.pytorch_backend.nets_utils import pad_list

from itertools import groupby
import editdistance
import random

CTC_LOSS_THRESHOLD = 10000

def time_mask(align, replace_value=0.0, num_mask=2, T=20, max_ratio=0.2):
    """Time masking

    :param torch.Tensor align: input tensor with shape (B, T, D)
    :param int replace_value: the value to be replaced with
    :param int T: maximum width of each mask
    :param int num_mask: number of masks
    """

    batch_size = align.size(0)
    len_align = align.size(1)
    max_len = int(max_ratio * len_align)
    current_len = 0
    mask = torch.full([batch_size, len_align], fill_value=True, dtype=torch.bool, device=align.device)

    for i in range(0, num_mask):
        t = random.randrange(0, T)
        t = min(t, max_len - current_len)
        t_zero = random.randrange(0, len_align - t)

        # avoids randrange error if values are equal and range is empty
        if t_zero == t_zero + t:
            return align, mask

        # mask_end = random.randrange(t_zero, t_zero + t)
        mask_end = t_zero + t
        align[:, t_zero:mask_end, :] = replace_value
        mask[:, t_zero:mask_end] = False

        current_len += t
    return align, mask

def repeat_after_batchdim1(x, n):
    return torch.reshape(x.unsqueeze(1).repeat(1, n), [n*x.size(0)])

def repeat_after_batchdim2(x, n):
    return torch.reshape(x.unsqueeze(1).repeat(1, n, 1), [n*x.size(0), x.size(1)])

def repeat_after_batchdim3(x, n):
    return torch.reshape(x.unsqueeze(1).repeat(1, n, 1, 1), [n*x.size(0), x.size(1), x.size(2)])

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

        # In case we use rnns for encoder.
        group.add_argument('--etype', default="", type=str,
                           help='Type of rnns to use in encoder')
        group.add_argument('--eprojs', default=320, type=int,
                           help='Number of encoder projection units')

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

        # Time masking
        group.add_argument('--decoder-num-mask', default=0, type=int,
                           help='Number of masks for decoder')
        group.add_argument('--decoder-mask-width', default=10, type=int,
                           help='Mask width for decoder')
        group.add_argument('--beta', default=0.0, type=float,
                           help='Weight of masked reconstruction loss')
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
        if not hasattr(args, "beta"):
            self.beta = 0.0
        else:
            self.beta = args.beta
        self.num_steps = args.num_steps
        self.tau = args.error_temperature
        self.idx_blank = 0
        self.masktoken = odim - 1
        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id
        self.lsm_weight = args.lsm_weight
        self.reporter = None

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

        # Define alignment embedding module.
        self.embed = torch.nn.Embedding(odim, self.enc_odim)
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
                src_attention_dropout_rate=args.transformer_attn_dropout_rate,
                input_layer="embed_linear",
                pos_enc_class=ScaledPositionalEncoding
            )

        # self.verbose = args.verbose
        self.reset_parameters(args)
        self.adim = args.adim

        # CTC directly works on top of encoder outputs.
        self.ctc = CTC(odim, self.enc_odim, args.dropout_rate, ctc_type=args.ctc_type, reduce=True)
        self.rnnlm = None

        # Decoder masking params
        self.decoder_num_mask = args.decoder_num_mask
        self.decoder_mask_width = args.decoder_mask_width

    def reset_parameters(self, args):
        """Initialize parameters."""
        # initialize parameters
        initialize(self, args.transformer_init)
        self.embed.weight.data.uniform_(-0.1, 0.1)

    def calculate_ters(self, ys_hat, hs_len, ys_pad, idx_blank=0, ignore_id=-1):
        """Calculate sentence-level CER score for CTC.

        :param torch.Tensor ys_hat: prediction (batch, seqlen)
        :param torch.Tensor hs_len: actual lengths of the input seqs (batch)
        :param torch.Tensor ys_pad: reference (batch, seqlen)
        :return: average sentence-level token errors
        :rtype float
        """
        errs, lens = [], []
        ys_pred = []
        for i, y in enumerate(ys_hat):
            # Weiran: groupby removes repetitions.
            y_hat = [x[0] for x in groupby(y[:hs_len[i]])]
            y_true = ys_pad[i]
            seq_hat, seq_true = [], []
            for idx in y_hat:
                idx = int(idx)
                if idx != ignore_id and idx != idx_blank:
                    seq_hat.append(idx)

            for idx in y_true:
                idx = int(idx)
                if idx != ignore_id and idx != idx_blank:
                    seq_true.append(idx)

            errs.append(editdistance.eval(seq_true, seq_hat))
            lens.append(len(seq_true))
            ys_pred.append(torch.tensor(seq_hat))

        return errs, lens, pad_list(ys_pred, self.ignore_id).to(ys_pad.device)

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

        B = self.num_steps

        # 1. forward encoder
        xs_pad = xs_pad[:, :max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)
        # Obtain the acoustic side features.
        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)

        batch_size = hs_pad.size(0)
        max_len_h = hs_pad.size(1)
        hs_len = hs_mask.view(batch_size, max_len_h).sum(1).to(torch.int32)

        if self.mtlalpha > 0:
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.enc_odim), hs_len, ys_pad)
        else:
            loss_ctc = 0.0

        # Forced alignment from ctc 0.
        mask_on_forcedalign = False
        if self.mtlalpha > 0 and mask_on_forcedalign:
            ctc_forcedalign = self.ctc.forced_alignment(hs_pad.view(batch_size, -1, self.enc_odim), hs_len, ys_pad).to(ys_pad.dtype)
        else:
            ctc_forcedalign = None

        # Initial greedy alignment.
        ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.enc_odim))
        err_ctc, _, ys_pred = self.calculate_ters(ys_hat.cpu(), hs_len.cpu(), ys_pad.cpu(), self.idx_blank, self.ignore_id)
        err_ctc = torch.tensor(err_ctc, dtype=hs_pad.dtype, device=hs_pad.device)

        # Auxiliary tensors for computing CTC loss.
        ys = [y[y != self.ignore_id] for y in ys_pad.cpu()]
        ys_len = torch.from_numpy(np.fromiter((x.size(0) for x in ys), dtype=np.int32))
        ys_true = torch.cat(ys).cpu().int()  # batch x olen

        # This is equivalent to having a <mask> token with fixed zero embedding.
        if self.training and self.decoder_num_mask > 0 and mask_on_forcedalign:
            forcedalign_onehot = torch.zeros([batch_size * max_len_h, self.odim], device=ys_hat.device)
            forcedalign_onehot = forcedalign_onehot.scatter(1, ctc_forcedalign.view(-1, 1), 1)
            forcedalign_onehot = forcedalign_onehot.reshape([batch_size, max_len_h, self.odim])
            forcedalign_onehot, mask = time_mask(forcedalign_onehot, num_mask=self.decoder_num_mask, T=self.decoder_mask_width)
            logprob_forcedalign = self.decoder(forcedalign_onehot, hs_mask, hs_pad, hs_mask)[0].log_softmax(dim=-1)
            """
            # Conditioning: for unmasked positions, use 1-hot probability.
            ctc_forcedalign = ctc_forcedalign.reshape([batch_size * max_len_h])
            logprob_forcedalign = logprob_forcedalign.reshape([batch_size * max_len_h, self.odim])
            unmasked_idx = torch.where(mask.reshape([-1]))[0]
            logprob_forcedalign[unmasked_idx, :] = - 1e10
            unmasked_labs = ctc_forcedalign[unmasked_idx]
            labs_idx = unmasked_idx * self.odim + unmasked_labs
            logprob_forcedalign.reshape([batch_size * max_len_h * self.odim])[labs_idx] = 0.0
            # Reshape back.
            logprob_forcedalign = logprob_forcedalign.reshape([batch_size, max_len_h, self.odim])
            """
            # ctc loss by dynamic programing.
            loss_masked = self.ctc.loss_fn(logprob_forcedalign.transpose(0, 1),
                                        ys_true, hs_len.cpu(), ys_len).to(logprob_forcedalign.device).sum()
        else:
            loss_masked = 0.0

        ########## Start the realigner part ##########
        loss_realigner = 0.0
        logprob = self.ctc.log_softmax(hs_pad)

        if dist.get_rank() == 0:
            logging.debug("ctc predictions:")
            logging.debug(ys_pred[0])
            logging.debug("initial ctc errors:")
            logging.debug(err_ctc)

        for i in range(B):
            if dist.get_rank() == 0:
                logging.debug("\n******** starting step %d ..." % i)

            # One-hot representation.
            if self.training and self.tau > 0:
                ys_hat_onehot = torch.nn.functional.gumbel_softmax(logprob, tau=self.tau, hard=True)
            else:
                ys_hat_onehot = torch.zeros([batch_size * max_len_h, self.odim], device=ys_hat.device)
                ys_hat_onehot = ys_hat_onehot.scatter(1, ys_hat.view(-1, 1), 1)
                ys_hat_onehot = ys_hat_onehot.reshape([batch_size, max_len_h, self.odim])

            # This is equivalent to having a <mask> token with fixed zero embedding.
            if self.training and self.decoder_num_mask > 0 and not mask_on_forcedalign:
                ys_hat_onehot, _ = time_mask(ys_hat_onehot, num_mask=self.decoder_num_mask, T=self.decoder_mask_width)

            # hs_mask is fixed throughout decoding.
            logprob = self.decoder(ys_hat_onehot, hs_mask, hs_pad, hs_mask)[0].log_softmax(dim=-1)
            tmp_loss = self.ctc.loss_fn(logprob.transpose(0, 1), ys_true, hs_len.cpu(), ys_len).to(logprob.device).sum()
            ys_hat = logprob.argmax(dim=-1)

            err_realigner, _, ys_pred = self.calculate_ters(ys_hat.cpu(), hs_len.cpu(), ys_pad.cpu(), self.idx_blank, self.ignore_id)
            err_realigner = torch.tensor(err_realigner, dtype=hs_pad.dtype, device=hs_pad.device)

            if dist.get_rank() == 0:
                logging.debug("Aligner CTC loss: %f" % float(tmp_loss))
                logging.debug("new label:")
                logging.debug(ys_pred[0])
                logging.debug("word errors:")
                logging.debug(err_realigner)

            loss_realigner += (1/B) * tmp_loss

        # total loss is the linear combination of two losses
        self.loss = self.mtlalpha * loss_ctc + (1 - self.mtlalpha) * loss_realigner + self.beta * loss_masked
        loss_ctc_data = float(loss_ctc)
        loss_realigner_data = float(loss_realigner)
        loss_masked_data = float(loss_masked)

        loss_data = float(self.loss)
        if loss_data < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            return self.loss, loss_data, loss_ctc_data, loss_realigner_data, loss_masked_data, err_ctc.cpu(), err_realigner.cpu()
        else:
            logging.warning('loss (=%f) is not correct', loss_data)
            return self.loss, loss_data, loss_ctc_data, loss_realigner_data, loss_masked_data, err_ctc.cpu(), err_realigner.cpu()

    def encode(self, x):
        """Encode acoustic features.

        :param ndarray x: source acoustic feature (T, D)
        :return: encoder outputs
        :rtype: torch.Tensor
        """
        self.eval()
        x = torch.as_tensor(x).unsqueeze(0)
        enc_output, _ = self.encoder(x, None)
        return enc_output.squeeze(0)

    def align2ys(self, align):
        y_hat = [x[0] for x in groupby(align)]
        yseq = []
        for idx in y_hat:
            idx = int(idx)
            if idx != self.ignore_id and idx != self.idx_blank:
                yseq.append(idx)
        return yseq

    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize input speech.
        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argument Namespace containing options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: N-best decoding results
        :rtype: list
        """

        if recog_args.input_context > 0:
            x = _context_concat(x, recog_args.input_context)
        if recog_args.input_skiprate > 1:
            x = x[0::recog_args.input_skiprate, :]

        hs_pad = self.encode(x).unsqueeze(0)
        T = hs_pad.size(1)
        hs_mask = make_non_pad_mask([T]).to(hs_pad.device).unsqueeze(-2)
        logging.info('input lengths: ' + str(T))

        # Weiran: this is CTC's greedy decoding.
        logprob = self.ctc.log_softmax(hs_pad.view(1, -1, self.enc_odim))
        ys_score, ys_hat = torch.max(logprob, dim=-1)
        yseq = self.align2ys(ys_hat[0])
        score = float(ys_score[0].sum())
        if recog_args.realigner_num_steps==0:
            nbest_hyps = []
            nbest_hyps.append({'yseq': [self.sos] + yseq + [self.eos], 'score': score})
            if char_list:
                hyp = "".join([char_list[y] for y in yseq])
                logging.info("prediction at step %d (score=%f): %s" % (0, score, hyp))
            return nbest_hyps

        batch_size=1
        realigner_use_sample=False
        NUM_SAMPLE=2

        if recog_args.realigner_num_steps < 0:
            # By default, use the steps with which the model is trained.
            num_steps = self.num_steps
        else:
            num_steps = recog_args.realigner_num_steps
        logging.info("number of decoding steps: %s" % num_steps)

        for i in range(num_steps):

            # Obtain new output for next round.
            if realigner_use_sample and self.tau > 0.0:
                if NUM_SAMPLE > 1:
                    batch_size *= NUM_SAMPLE
                    logprob = repeat_after_batchdim3(logprob, NUM_SAMPLE)
                    hs_pad = repeat_after_batchdim3(hs_pad, NUM_SAMPLE)
                    hs_mask = repeat_after_batchdim3(hs_mask, NUM_SAMPLE)
                ys_hat_onehot = torch.nn.functional.gumbel_softmax(logprob, tau=self.tau, hard=True)
            else:
                ys_hat_onehot = torch.zeros([batch_size * T, self.odim], device=ys_hat.device)
                ys_hat_onehot = ys_hat_onehot.scatter(1, ys_hat.view(-1, 1), 1)
                ys_hat_onehot = ys_hat_onehot.reshape([batch_size, T, self.odim])

            # hs_mask is fixed throughout decoding.
            logprob = self.decoder(ys_hat_onehot, hs_mask, hs_pad, hs_mask)[0].log_softmax(dim=-1)

            ys_score, ys_hat = torch.max(logprob, dim=-1)
            yseq = self.align2ys(ys_hat[0])
            score = float(ys_score[0].sum())
            if char_list:
                hyp = "".join([char_list[y] for y in yseq])
                logging.info("prediction at step %d (score=%f): %s" % (i, score, hyp))

        ys_score, ys_hat = torch.max(logprob.mean(dim=0, keepdim=True), dim=-1)
        yseq = self.align2ys(ys_hat[0])
        score = float(ys_score[0].sum())
        if char_list:
            hyp = "".join([char_list[y] for y in yseq])
            logging.info("prediction at step %d (score=%f): %s" % (i, score, hyp))

        nbest_hyps = []
        nbest_hyps.append({'yseq': [self.sos] + yseq + [self.eos], 'score': float(ys_score[0].sum())})

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
