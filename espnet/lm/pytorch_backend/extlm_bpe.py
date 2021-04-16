#!/usr/bin/env python3

# Copyright 2018 Mitsubishi Electric Research Laboratories (Takaaki Hori)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.lm.lm_utils import make_lexical_tree
from espnet.lm.lm_utils import make_lexical_tree_with_lexicon
from espnet.nets.pytorch_backend.nets_utils import to_device

import logging

# Definition of a multi-level (subword/word) language model
class MultiLevelLM_with_lexicon(nn.Module):
    logzero = -10000000000.0
    zero = 1.0e-10

    def __init__(self, wordlm, subwordlm, word_dict, subword_dict, lexicon_dict, space_char,
                 subwordlm_weight=0.8, oov_penalty=0.001, open_vocab=False):

        super(MultiLevelLM_with_lexicon, self).__init__()
        self.wordlm = wordlm
        self.subwordlm = subwordlm
        self.word_dict = word_dict
        self.subword_dict = subword_dict
        self.word_eos = word_dict['<eos>']
        self.word_unk = word_dict['<unk>']
        self.var_word_eos = torch.LongTensor([self.word_eos])
        self.var_word_unk = torch.LongTensor([self.word_unk])
        self.incomplete_word = -99
        self.index_to_subword = dict([(subword_dict[k], k) for k in subword_dict])
        self.index_to_word = dict([(word_dict[k], k) for k in word_dict])

        # BPE's word delimiter.
        self.fn_test_idx_space = lambda x: self.index_to_subword[x].startswith(space_char)
        self.eos = subword_dict['<eos>']
        # self.lexroot = make_lexical_tree(word_dict, subword_dict, self.word_unk)
        self.lexroot = make_lexical_tree_with_lexicon(word_dict, subword_dict, self.word_unk, self.word_eos, lexicon_dict)
        self.log_oov_penalty = math.log(oov_penalty)
        self.open_vocab = open_vocab
        self.subword_dict_size = len(subword_dict)
        self.subwordlm_weight = subwordlm_weight
        self.normalized = True

    def forward(self, state, x):
        # update state with input label x
        if state is None:  # make initial states and log-prob vectors
            # TODO: (Weiran) should eos be sos?
            # Weiran: my guess is that x here can only be <sos> in the subword dict.
            logging.info("MultiLevelLM state=None. The token being accepted is %d" % x)
            self.var_word_eos = to_device(self, self.var_word_eos)
            self.var_word_unk = to_device(self, self.var_word_unk)
            wlm_state, z_wlm = self.wordlm(None, self.var_word_eos)
            wlm_logprobs = F.log_softmax(z_wlm, dim=1)
            clm_state, z_clm = self.subwordlm(None, x)
            clm_logprobs = F.log_softmax(z_clm, dim=1)
            # log_y is the lookahead probability. The scores computed here for tokens are added in the NEXT beam search step.
            log_y = clm_logprobs * self.subwordlm_weight
            new_node = self.lexroot
            # Accumulating clm probability until seeing a new word, at which point acc_subword_logprob resets to 0.
            acc_subword_logprob = 0.
            return [((clm_state, clm_logprobs, wlm_state, wlm_logprobs, new_node, float(acc_subword_logprob)), log_y, self.incomplete_word)]
        else:
            clm_state, clm_logprobs, wlm_state, wlm_logprobs, node, acc_subword_logprob = state
            xi = int(x)
            # Weiran: the subword <eos> never occurs in the lexical tree.
            # assert not xi==self.eos, "xi can be eos ???!!!"

            if self.fn_test_idx_space(xi) and (not node==self.lexroot):  # inter-word transition
                # Weiran: In this case, we need to update wordlm state.
                if node is not None and len(node[1]) > 0:  # check if the node is word end
                    # Weiran: now we can have multiple words in the same node.
                    WIDS = [to_device(self, torch.LongTensor([wid])) for wid in node[1]]
                else:  # this node is not a word end, which means <unk>
                    WIDS = [self.var_word_unk]

                RET = []
                # update wordlm state and log-prob vector
                for wid in WIDS:
                    # Weiran: I am absorbing the word-lm score into log_y.

                    if wid == self.var_word_unk:
                        tmp_wlm_logprob = wlm_logprobs[:, self.word_unk] + self.log_oov_penalty
                    else:
                        logging.info("(forward) appending the word %s with score %f, will remove clm score %f" %
                              (self.index_to_word[int(wid)], wlm_logprobs[:, wid], float(acc_subword_logprob)))
                        # Weiran (04/15): fixed (self.index_to_word[int(wid)], wlm_logprobs[:, wid], float(acc_subword_logprob - clm_logprobs[:, xi] * self.subwordlm_weight)))
                        # Weiran (04/15): fixed tmp_wlm_logprob = wlm_logprobs[:, wid] - acc_subword_logprob + clm_logprobs[:, xi] * self.subwordlm_weight
                        tmp_wlm_logprob = wlm_logprobs[:, wid] - acc_subword_logprob

                    # Move in word lm state.
                    new_wlm_state, new_z_wlm = self.wordlm(wlm_state, wid)
                    new_wlm_logprobs = F.log_softmax(new_z_wlm, dim=1)
                    new_node = self.lexroot  # move to the tree root
                    # After seeing a word, reset the acc_subword_logprob.
                    new_acc_subword_logprob = clm_logprobs[:, xi] * self.subwordlm_weight

                    # Weiran: we just finished handling the previously finished word.
                    # And now lexroot meets a word start, we need to move one more step in the lexicon tree.
                    new_node = new_node[0][xi]
                    new_clm_state, new_z_clm = self.subwordlm(clm_state, x)
                    new_clm_logprobs = F.log_softmax(new_z_clm, dim=1)

                    # Weiran: This is the lookahead probability.
                    # I am incorporating the word language model scores here.
                    # No matter which token is accepted next, the lm score for the just completed word is added.
                    new_log_y = tmp_wlm_logprob + new_clm_logprobs * self.subwordlm_weight
                    RET.append(((new_clm_state, new_clm_logprobs, new_wlm_state, new_wlm_logprobs, new_node, float(new_acc_subword_logprob)), new_log_y, int(wid)))

                return RET
            else:
                # The BPE token is not a beginning of word.
                if node is not None and xi in node[0]:  # intra-word transition
                    new_node = node[0][xi]

                    logging.info("(forward) accumulating acc_subword_logprob %f by adding score %f from %s (%d)" %
                          (acc_subword_logprob, clm_logprobs[0, xi] * self.subwordlm_weight, self.index_to_subword[xi], xi))
                    new_acc_subword_logprob = acc_subword_logprob + clm_logprobs[0, xi] * self.subwordlm_weight

                    new_clm_state, new_z_clm = self.subwordlm(clm_state, x)
                    new_clm_logprobs = F.log_softmax(new_z_clm, dim=1)
                    new_log_y = new_clm_logprobs * self.subwordlm_weight
                elif self.open_vocab:  # if no path in the tree, enter open-vocabulary mode (node will stay in none and only subword language model scores are used afterwards.
                    logging.info("Entering open vocab mode!!!")
                    new_node = None
                    new_acc_subword_logprob = acc_subword_logprob + clm_logprobs[0, xi] * self.subwordlm_weight

                    new_clm_state, new_z_clm = self.subwordlm(clm_state, x)
                    new_clm_logprobs = F.log_softmax(new_z_clm, dim=1)
                    new_log_y = new_clm_logprobs * self.subwordlm_weight
                else:  # if open_vocab flag is disabled, return 0 probabilities.
                    logging.info("Killing the hypothesis!!!")
                    # Weiran: this hypothesis is effectively killed.
                    new_log_y = to_device(self, torch.full((1, self.subword_dict_size), self.logzero))
                    return [((clm_state, clm_logprobs, wlm_state, wlm_logprobs, None, 0.), new_log_y, self.incomplete_word)]

                # Note that word language model does not forward.
                return [((new_clm_state, new_clm_logprobs, wlm_state, wlm_logprobs, new_node, float(new_acc_subword_logprob)), new_log_y, self.incomplete_word)]


    def final(self, state):

        # First get the word at current lexical node.
        clm_state, clm_logprobs, wlm_state, wlm_logprobs, node, acc_subword_logprob = state
        if node is not None and len(node[1]) > 0:  # check if the node is word end
            # Weiran: we can have multiple words in the same node.
            WIDS = [to_device(self, torch.LongTensor([wid])) for wid in node[1]]
        else:  # this node is not a word end, which means <unk>
            WIDS = [self.var_word_unk]

        RET = []
        # update wordlm state and log-prob vector.
        for wid in WIDS:
            new_wlm_state, new_z_wlm = self.wordlm(wlm_state, wid)

            if wid == self.var_word_unk:
                tmp_wlm_logprob = wlm_logprobs[:, self.word_unk] + self.log_oov_penalty
            else:
                tmp_wlm_logprob = wlm_logprobs[:, wid] - acc_subword_logprob

            # Need to remove the token-level <eos> score and add word-level <eos> score.
            logging.info("(final) appending the word %s with score %f, will remove clm score (including eos) %f" % (self.index_to_word[int(wid)], wlm_logprobs[:, wid], acc_subword_logprob))
            logging.info("(final) adding word_eos_score=%f" % (float(F.log_softmax(new_z_wlm, dim=1)[:, self.word_eos])))

            final_score = tmp_wlm_logprob + float(F.log_softmax(new_z_wlm, dim=1)[:, self.word_eos])

            RET.append((float(final_score), int(wid)))

        return RET


# Definition of a look-ahead word language model
class LookAheadWordLM(nn.Module):
    logzero = -10000000000.0
    zero = 1.0e-10

    def __init__(self, wordlm, word_dict, subword_dict, oov_penalty=0.0001, open_vocab=True):
        super(LookAheadWordLM, self).__init__()
        self.wordlm = wordlm
        self.word_eos = word_dict['<eos>']
        self.word_unk = word_dict['<unk>']
        self.var_word_eos = torch.LongTensor([self.word_eos])
        self.var_word_unk = torch.LongTensor([self.word_unk])
        self.space = subword_dict['<space>']
        self.eos = subword_dict['<eos>']
        self.lexroot = make_lexical_tree(word_dict, subword_dict, self.word_unk)
        self.oov_penalty = oov_penalty
        self.open_vocab = open_vocab
        self.subword_dict_size = len(subword_dict)
        self.zero_tensor = torch.FloatTensor([self.zero])
        self.normalized = True

    def forward(self, state, x):
        # update state with input label x
        if state is None:  # make initial states and cumlative probability vector
            self.var_word_eos = to_device(self, self.var_word_eos)
            self.var_word_unk = to_device(self, self.var_word_unk)
            self.zero_tensor = to_device(self, self.zero_tensor)
            wlm_state, z_wlm = self.wordlm(None, self.var_word_eos)
            cumsum_probs = torch.cumsum(F.softmax(z_wlm, dim=1), dim=1)
            new_node = self.lexroot
            xi = self.space
        else:
            wlm_state, cumsum_probs, node = state
            xi = int(x)
            if xi == self.space:  # inter-word transition
                if node is not None and node[1] >= 0:  # check if the node is word end
                    w = to_device(self, torch.LongTensor([node[1]]))
                else:  # this node is not a word end, which means <unk>
                    w = self.var_word_unk
                # update wordlm state and cumulative probability vector
                wlm_state, z_wlm = self.wordlm(wlm_state, w)
                cumsum_probs = torch.cumsum(F.softmax(z_wlm, dim=1), dim=1)
                new_node = self.lexroot  # move to the tree root
            elif node is not None and xi in node[0]:  # intra-word transition
                new_node = node[0][xi]
            elif self.open_vocab:  # if no path in the tree, enter open-vocabulary mode
                new_node = None
            else:  # if open_vocab flag is disabled, return 0 probabilities
                log_y = to_device(self, torch.full((1, self.subword_dict_size), self.logzero))
                return (wlm_state, None, None), log_y

        if new_node is not None:
            succ, wid, wids = new_node
            # compute parent node probability
            sum_prob = (cumsum_probs[:, wids[1]] - cumsum_probs[:, wids[0]]) if wids is not None else 1.0
            if sum_prob < self.zero:
                log_y = to_device(self, torch.full((1, self.subword_dict_size), self.logzero))
                return (wlm_state, cumsum_probs, new_node), log_y
            # set <unk> probability as a default value
            unk_prob = cumsum_probs[:, self.word_unk] - cumsum_probs[:, self.word_unk - 1]
            y = to_device(self, torch.full((1, self.subword_dict_size), float(unk_prob) * self.oov_penalty))
            # compute transition probabilities to child nodes
            for cid, nd in succ.items():
                y[:, cid] = (cumsum_probs[:, nd[2][1]] - cumsum_probs[:, nd[2][0]]) / sum_prob
            # apply word-level probabilities for <space> and <eos> labels
            if wid >= 0:
                wlm_prob = (cumsum_probs[:, wid] - cumsum_probs[:, wid - 1]) / sum_prob
                y[:, self.space] = wlm_prob
                y[:, self.eos] = wlm_prob
            elif xi == self.space:
                y[:, self.space] = self.zero
                y[:, self.eos] = self.zero
            log_y = torch.log(torch.max(y, self.zero_tensor))  # clip to avoid log(0)
        else:  # if no path in the tree, transition probability is one
            log_y = to_device(self, torch.zeros(1, self.subword_dict_size))
        return (wlm_state, cumsum_probs, new_node), log_y

    def final(self, state):
        wlm_state, cumsum_probs, node = state
        if node is not None and node[1] >= 0:  # check if the node is word end
            w = to_device(self, torch.LongTensor([node[1]]))
        else:  # this node is not a word end, which means <unk>
            w = self.var_word_unk
        wlm_state, z_wlm = self.wordlm(wlm_state, w)
        return float(F.log_softmax(z_wlm, dim=1)[:, self.word_eos])
