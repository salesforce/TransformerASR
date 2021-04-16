#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# Copyright 2019 Shigeki Karita
# Copyright 2020 Weiran Wang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Subsampling layer definition."""

import numpy as np
import torch

from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding

def _context_concat(seq, context_size=0):
    """ seq is of size length x feat_dim.
    output is of size length x (feat_dim*(1+2*context_size)).
    """

    if context_size == 0:
        return seq

    output = []
    length = seq.size(0)
    # Left concatenation.
    for j in range(context_size):
        tmp = torch.cat([seq[0:1, :].repeat([j + 1, 1]), seq[0:(length - j - 1), :]], dim=0)
        output.append(tmp)

    # Add original inputs.
    output.append(seq)

    # Right concatenation.
    for j in range(context_size):
        tmp = torch.cat([seq[(j + 1):length, :], seq[length-1:length, :].repeat([j + 1, 1])], dim=0)
        output.append(tmp)

    return torch.cat(output, dim=1)


def _context_concat_numpy(seq, context_size=0):
    """ seq is of size length x feat_dim.
    output is of size length x (feat_dim*(1+2*context_size)).
    """

    if context_size == 0:
        return seq

    output = []
    length = seq.shape[0]
    # Left concatenation.
    for j in range(context_size):
        tmp = np.concatenate([np.repeat(seq[np.newaxis, 0, :], j + 1, axis=0), seq[0:(length - j - 1), :]], 0)
        output.append(tmp)

    # Add original inputs.
    output.append(seq)

    # Right concatenation.
    for j in range(context_size):
        tmp = np.concatenate([seq[(j + 1):length, :], np.repeat(seq[np.newaxis, length - 1, :], j + 1, axis=0)], 0)
        output.append(tmp)

    return np.concatenate(output, 1)


class Conv2dSubsampling(torch.nn.Module):
    """Convolutional 2D subsampling (to 1/4 length).

    :param int idim: input dim
    :param int odim: output dim
    :param float dropout_rate: dropout rate

    """

    def __init__(self, idim, odim, dropout_rate):
        """Construct an Conv2dSubsampling object."""
        super(Conv2dSubsampling, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(1, odim, 3, 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(odim, odim, 3, 2),
            torch.nn.ReLU()
        )
        self.out = torch.nn.Sequential(
            torch.nn.Linear(odim * (((idim - 1) // 2 - 1) // 2), odim),
            PositionalEncoding(odim, dropout_rate)
        )

    def forward(self, x, x_mask):
        """Subsample x.

        :param torch.Tensor x: input tensor
        :param torch.Tensor x_mask: input mask
        :return: subsampled x and mask
        :rtype Tuple[torch.Tensor, torch.Tensor]
        """
        x = x.unsqueeze(1)  # (b, c, t, f)
        x = self.conv(x)
        b, c, t, f = x.size()
        x = self.out(x.transpose(1, 2).contiguous().view(b, t, c * f))
        if x_mask is None:
            return x, None
        if x_mask.size(1)==1:
            return x, x_mask[:, :, :-2:2][:, :, :-2:2]
        else:
            # Weiran: if the mask is full, both time dimensions need to be subsampled.
            return x, x_mask[:, :-2:2, :-2:2][:, :-2:2, :-2:2]
