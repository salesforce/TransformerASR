"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

from argparse import Namespace

import logging

import torch
import json
import numpy
import math
from espnet.nets.pytorch_backend.transformer.subsampling import _context_concat

from speech_datasets import SpeechDataLoader
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.asr.pytorch_backend.asr_init import load_trained_model
from espnet.asr.asr_utils import get_model_conf
import espnet.nets.pytorch_backend.lm.default as lm_pytorch
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.pytorch_backend.asr_dsl import _recursive_to, CustomConverter

NEG_INF = - 999999999.9

def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.
    """

    set_deterministic_pytorch(args)
    # Weiran: the model shall be trained with certain left context and right context.
    model, train_args = load_trained_model(args.model)
    assert isinstance(model, ASRInterface)
    model.recog_args = args

    # read rnnlm
    if args.rnnlm:
        rnnlm_args = get_model_conf(args.rnnlm, args.rnnlm_conf)
        if getattr(rnnlm_args, "model_module", "default") != "default":
            raise ValueError("use '--api v2' option to decode with non-default language model")
        rnnlm = lm_pytorch.ClassifierWithState(
            lm_pytorch.RNNLM(len(train_args.char_list), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.rnnlm, rnnlm)
        rnnlm.eval()
    else:
        rnnlm = None

    # Read truth file.
    if args.truth_file:
        with open(args.truth_file, "r") as fin:
            retval = fin.read().rstrip('\n')
            dict_truth = dict([(l.split()[0], " ".join(l.split()[1:])) for l in retval.split("\n")])

    """
    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()
    """

    new_js = {}
    recog_converter = CustomConverter(input_context=args.input_context, input_skiprate=args.input_skiprate,
                                      mode="eval", dtype=torch.float32)
    recog_data = list(filter(lambda s: len(s) > 0, map(lambda s: s.strip(), args.recog_sets.split(","))))
    recog_loader = SpeechDataLoader(recog_data, task="asr", shuffle=False,
                                    precomputed_feats_type=args.precomputed_feats_type,
                                    batch_size=1, spmodel=args.spmodel, token_list=train_args.char_list,
                                    transform_conf=args.preprocess_conf, train=False, num_workers=args.loader_num_worker,
                                    data_cache_mb=args.loader_cache_mb, num_replicas=args.num_replicas, rank=args.jobid-1,
                                    ensure_equal_parts=False, text_filename=args.text_filename)

    with torch.no_grad():
        idx = 0
        for batch in recog_loader:
            idx += 1
            name = batch[0]['uttid']
            logging.info('(%d/%d) decoding ' + name, idx, len(recog_loader))
            feat = _recursive_to(recog_converter(batch), device=torch.device('cpu'))[0]
            feat = feat[0]

            if args.truth_file and name in dict_truth:
                truth_text = dict_truth[name]
            else:
                truth_text = ""

            nbest_hyps = recognize_online(model, feat, args, train_args.char_list, rnnlm, truth_text=truth_text, blankidx=0)

            # Weiran: prepare dict in order to add decoding results. Skipped the input and shape information.
            gt_tokens = [int(_) for _ in batch[0]["labels"]]
            tmp_dict = {"output": [{"name": "target1", "text": batch[0]["text"],
                                    "tokenid": " ".join([str(_) for _ in gt_tokens]).strip(),
                                    "token": " ".join([train_args.char_list[_] for _ in gt_tokens]).strip()}],
                                    "utt2spk": batch[0]["speaker"]}

            # Weiran: I am adding text in words in the result json.
            new_js[name] = add_results_to_json(tmp_dict, nbest_hyps, train_args.char_list, copy_times=True)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))



def logsumexp(*args):
    """
    Stable log sum exp.
    """
    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a - a_max) for a in args))
    return a_max + lsp



def recognize_online(model, input, recog_args, char_list, rnnlm=None, truth_text="", blankidx=0):
    """Recognize input speech.
    :param model: the trained hybrid att+CTC model.
    :param ndnarray input: input acoustic feature (T, D)
    :param Namespace recog_args: argument Namespace containing options
    :param list char_list: list of characters
    :param int blankidx: the index of the CTC <blank> symbol
    :param torch.nn.Module rnnlm: language model module
    :param string truth_text: ground truth text for the utterance
    :return: N-best decoding results
    :rtype: list
    """

    # Assuming this is a streaming-compliant encoder.
    enc_output = model.encode(input).unsqueeze(0)

    # Use ctc beams to obtain timing for attention model.
    lpz = model.ctc.log_softmax(enc_output)
    lpz = lpz.detach().numpy().squeeze(0)

    # Max time.
    T = lpz.shape[0]
    S = lpz.shape[1]
    logging.info('input lengths: %s' % T)
    logging.info('vocab size: %s' % S)
    if truth_text:
        logging.info("Ground truth in words:")
        logging.info(truth_text)

    # Search parameters.
    beam_size = recog_args.beam_size
    penalty = recog_args.penalty
    ctc_weight = recog_args.ctc_weight
    ctc_max_active = recog_args.streaming_ctc_maxactive
    blank_offset = recog_args.streaming_ctc_blank_offset

    # Some new search parameters not existing in offline decoding.
    beam_width = recog_args.streaming_beam_width
    # att_delay is the estimate of attention span of each output token.
    # att_delay can be set to the value used for CTC-triggered attention training.
    att_delay = recog_args.streaming_att_delay
    # Weiran: I am now allowing some bounded delay for decoder, this could be different from att_delay.
    dec_delay = recog_args.streaming_dec_delay

    # The initial CTC-prefix search phase uses a larger beamsize.
    ctc_scoring_ratio = recog_args.streaming_scoring_ratio

    # Prepare sos and eos.
    eos_idx = model.eos
    y = model.sos
    vy = enc_output.new_zeros(1).long()
    vy[0] = y

    # Initialize CTC hypothesis.
    rnnlm_state, rnnlm_scores = rnnlm.predict(None, vy)
    rnn_num_layers = len(rnnlm_state['c'])
    rnnlm_scores = rnnlm_scores.squeeze(0).numpy()
    # ctc_scores are probabilities of prefix ending with blank and non_blank.
    # prev_score tracks the score right before a new token is added.
    hyp_ctc = {'ctc_scores': (0.0, NEG_INF), 'prev_score': 0.0,
               'accum_lmscore': 0.0, 'rnnlm_state': rnnlm_state, 'rnnlm_scores': rnnlm_scores, 'ctc_times': [-1]}

    # Initialize att hypothesis.
    # The name cache is consistent with that for attention layers.
    hyp_att = {'att_score': 0.0, 'att_score_last': 0.0, 'cache': None, 'att_times': [-1]}

    # We will maintain two set of hypothesis, for the att and ctc model respectively.
    # Use dict for hypothesis sets so that it is efficient to determine certain hypo appears in each set.
    hyps_ctc = dict()
    hyps_att = dict()
    # There is only one partial hypothesis for now.
    l = tuple([model.sos])
    hyps_ctc[l] = hyp_ctc
    hyps_att[l] = hyp_att

    # Perform frame synchronous decoding for CTC. t is index for frame.
    for t in range(T):  # Loop over time.

        logging.debug("\n")
        logging.debug("=" * 80)
        logging.debug("CTC beam search at frame %d ..." % t)
        hyps_new = dict()

        # CTC-PREFIX beam search.
        # Logic is mostly taken from https://gist.github.com/awni/56369a90d03953e370f3964c826ed4b0
        active_set = numpy.argpartition(lpz[t], -ctc_max_active)[-ctc_max_active:]
        for s in active_set:
            p = lpz[t, s]
            # logging.debug("logprob for token %d (%s): %f" % (s, char_list[s], p))
            for l in hyps_ctc:
                p_b, p_nb = hyps_ctc[l]['ctc_scores']
                prev_score = hyps_ctc[l]['prev_score']
                rnnlm_state = hyps_ctc[l]['rnnlm_state']
                rnnlm_scores = hyps_ctc[l]['rnnlm_scores']
                accum_lmscore = hyps_ctc[l]['accum_lmscore']
                ctc_times = hyps_ctc[l]['ctc_times']

                if s==blankidx:  # blank. Not outputing new token.
                    p = p + blank_offset
                    if l in hyps_new:
                        p_b_new, p_nb_new = hyps_new[l]['ctc_scores']
                    else:
                        p_b_new, p_nb_new = NEG_INF, NEG_INF
                    # Only the probability of ending in blank gets updated.
                    ctc_scores_new = (logsumexp(p_b_new, p_b + p, p_nb + p), p_nb_new)
                    hyps_new[l] = {'ctc_scores': ctc_scores_new, 'prev_score': prev_score,
                                   'accum_lmscore': accum_lmscore,
                                   'rnnlm_state': rnnlm_state, 'rnnlm_token': -1, 'rnnlm_scores': rnnlm_scores,
                                   'ctc_times': list(ctc_times)}
                else:
                    if s == l[-1]:
                        # CTC does not model the acoustic scores of <sos> and <eos>.
                        # I doubt this can happen when active_set does not contain everything.
                        if len(l)==1:
                            continue

                        # Choice 1: Not outputing new token.
                        if l in hyps_new:
                            p_b_new, p_nb_new = hyps_new[l]['ctc_scores']
                        else:
                            p_b_new, p_nb_new = NEG_INF, NEG_INF
                        ctc_scores_new = (p_b_new, logsumexp(p_nb_new, p_nb + p))
                        hyps_new[l] = {'ctc_scores': ctc_scores_new, 'prev_score': prev_score,
                                       'accum_lmscore': accum_lmscore,
                                       'rnnlm_state': rnnlm_state, 'rnnlm_token': -1, 'rnnlm_scores': rnnlm_scores,
                                       'ctc_times': list(ctc_times)}

                        # Choice 2: Outputing new token.
                        newl = l + (s,)
                        if newl in hyps_new:
                            p_b_new, p_nb_new = hyps_new[newl]['ctc_scores']
                            ctc_times = list(hyps_new[newl]['ctc_times'])
                        else:
                            p_b_new, p_nb_new = NEG_INF, NEG_INF
                            ctc_times = ctc_times + [t]
                        ctc_scores_new = (p_b_new, logsumexp(p_nb_new, p_b + p))
                        hyps_new[newl] = {'ctc_scores': ctc_scores_new, 'prev_score': logsumexp(p_b, p_nb),
                                          'accum_lmscore': accum_lmscore + rnnlm_scores[s],
                                          'rnnlm_state': rnnlm_state, 'rnnlm_token': s, 'rnnlm_scores': rnnlm_scores,
                                          'ctc_times': ctc_times}
                    else:
                        # Must output new token.
                        newl = l + (s,)
                        if newl in hyps_new:
                            p_b_new, p_nb_new = hyps_new[newl]['ctc_scores']
                            ctc_times = list(hyps_new[newl]['ctc_times'])
                        else:
                            p_b_new, p_nb_new = NEG_INF, NEG_INF
                            ctc_times = ctc_times + [t]
                        ctc_scores_new = (p_b_new, logsumexp(p_nb_new, p_b + p, p_nb + p))
                        hyps_new[newl] = {'ctc_scores': ctc_scores_new, 'prev_score': logsumexp(p_b, p_nb),
                                          'accum_lmscore': accum_lmscore + rnnlm_scores[s],
                                          'rnnlm_state': rnnlm_state, 'rnnlm_token': s, 'rnnlm_scores': rnnlm_scores,
                                          'ctc_times': ctc_times}
            # End of loop over active set.
        # End of loop over hyps.

        # Sort and trim the beams for one time step.
        # First dictionary to list.
        hyps_list = [(k, hyps_new[k]) for k in hyps_new]
        del hyps_new
        # Check the equation for computing total score for pruning.
        hyps_list = sorted(hyps_list,
            key=lambda x: logsumexp(*x[1]['ctc_scores']) + recog_args.lm_weight * x[1]['accum_lmscore'] + penalty * (len(x[0])-1),
            reverse=True)[:(beam_size * ctc_scoring_ratio)]
        # Get total score for top beam.
        score_thres = \
            logsumexp(*hyps_list[0][1]['ctc_scores']) + recog_args.lm_weight * hyps_list[0][1]['accum_lmscore'] + penalty * (len(hyps_list[0][0])-1)
        # Remove hyps whose total score is below that of top beam by beam_width.
        hyps_list = list(filter(lambda x: score_thres-beam_width <=
            logsumexp(*x[1]['ctc_scores']) + recog_args.lm_weight * x[1]['accum_lmscore'] + penalty * (len(x[0])-1), hyps_list))
        # Back up the top beams for JOINT. Note that we do not have ctc_scoring_ratio here.
        ctc_top_keys = set([k for (k, v) in hyps_list[:min(len(hyps_list), beam_size)]])

        # Batch-update the rnnlm_state and rnnlm_scores for hyps whose rnnlm_token are non-negative.
        tokens_to_update = [v['rnnlm_token'] for (k, v) in hyps_list if v['rnnlm_token']>=0]
        if tokens_to_update:
            # Each rnnlm_state is a dict, so this is a list of dict.
            # Each dict has 'c' and 'h', each state is a list of tensors over layers.
            states_to_update = [v['rnnlm_state'] for (k, v) in hyps_list if v['rnnlm_token'] >= 0]
            tmp_states = dict()
            # First group samples, then group layers.
            tmp_states['c'] = [torch.cat([ss['c'][i] for ss in states_to_update], 0) for i in range(rnn_num_layers)]
            tmp_states['h'] = [torch.cat([ss['h'][i] for ss in states_to_update], 0) for i in range(rnn_num_layers)]
            logging.debug("\nForwarding rnnlm of %d samples" % len(tokens_to_update))
            new_rnnlm_states, new_rnnlm_scores = rnnlm.predict(tmp_states, torch.tensor(tokens_to_update).long())
        hyps_ctc = dict()
        tmp_count = 0
        for k, v in hyps_list:
            if v['rnnlm_token'] >= 0:
                v['rnnlm_state'] = {'c': [new_rnnlm_states['c'][i][tmp_count, :].unsqueeze(0) for i in range(rnn_num_layers)],
                                    'h': [new_rnnlm_states['h'][i][tmp_count, :].unsqueeze(0) for i in range(rnn_num_layers)]}
                v['rnnlm_scores'] = new_rnnlm_scores[tmp_count, :].cpu()
                tmp_count += 1
            v.pop('rnnlm_token', None)
            hyps_ctc[k] = v

        logging.debug("\nFrame %d, CTCPrefix finished ..." % t)
        for l in list(hyps_ctc.keys())[:min(10, len(hyps_ctc))]:
            logging.debug("hyp: [%s], total score: %f" % ("".join([char_list[x] for x in l]),
                logsumexp(*hyps_ctc[l]['ctc_scores']) + recog_args.lm_weight * hyps_ctc[l]['accum_lmscore'] + penalty * (len(l)-1)))

        # Weiran: this is the DCOND step in the paper.
        ys = []
        tokens_to_update = []
        caches = []
        for l in hyps_ctc:
            if len(l)>1:
                c = l[-1]
                ctc_time = hyps_ctc[l]['ctc_times'][-1]
                if t - ctc_time > 2 and lpz[t, c] > math.log(0.01) > max(lpz[ctc_time+1, c], lpz[ctc_time+2, c]):
                    # Weiran: I am also updating the ctc_times.
                    logging.debug("for hyp [%s], changing ctc_time from %d to %d" %
                          ("".join([char_list[x] for x in l]), ctc_time, t))
                    hyps_ctc[l]['ctc_times'][-1] = t
                    if l in hyps_att:  # and hyps_att[l]['att_times'][-1] < t-2:
                        ys.append(l[:-1])
                        tokens_to_update.append(c)
                        caches.append([ca[:-1,:] for ca in hyps_att[l]['cache']])

        num_to_update = len(ys)
        if num_to_update > 0:
            logging.debug("Adjusting %d att hyps ..." % num_to_update)
            # Memory up to frame t.
            hs_pad = enc_output[:, :min(t+1+dec_delay, T), :].repeat([num_to_update, 1, 1])
            local_att_scores, new_caches = model.decoder.batch_forward_one_step_with_cache(ys, caches, hs_pad, None)
            local_att_scores = local_att_scores.numpy()

            for i in range(num_to_update):
                l = ys[i] + (tokens_to_update[i],)
                logging.debug("Attention: Appending %s to [%s]" %
                      (char_list[tokens_to_update[i]], " ".join([char_list[_] for _ in ys[i]])))
                logging.debug("att_score_last changes from %f to %f" %
                      (hyps_att[l]['att_score_last'], local_att_scores[i, tokens_to_update[i]]))
                hyps_att[l]['att_score'] = hyps_att[l]['att_score'] - hyps_att[l]['att_score_last'] + local_att_scores[i, tokens_to_update[i]]
                hyps_att[l]['att_score_last'] = local_att_scores[i, tokens_to_update[i]]
                hyps_att[l]['cache'] = new_caches[i]
                hyps_att[l]['att_times'][-1] = min(t+dec_delay, T-1)
                # If some ctc new hyps can be premature, the actual time delay can be too short??
            logging.debug("Computing attention scores finished ...")
            del new_caches
        logging.debug("\n")

        # Compute attention scores, which lags behind CTC by at most 1 token.
        logging.debug("\n")
        logging.debug("<" * 40)
        logging.debug("Attention pass I: checking hypothesis ...")
        ys = []
        tokens_to_update = []
        caches = []
        for l in hyps_ctc:
            if l not in hyps_att:
                # logging.debug("Attention: Considering augmenting the hyp: [%s]" % " ".join([char_list[_] for _ in l]))
                oldl = l[:-1]
                if oldl in hyps_att:
                    c = l[-1]
                    ctc_time = hyps_ctc[l]['ctc_times'][-1]
                    # If it has been sufficient number of frames since CTC added the token c.
                    if t + dec_delay - ctc_time >= att_delay:
                        assert t + dec_delay - ctc_time == att_delay, "weird timing !"
                        logging.debug("Attention: Will append %s to [%s]" %
                              (char_list[c], " ".join([char_list[_] for _ in oldl])))
                        logging.debug("Attention: Perfect delay %d (current t=%d, ctc_time=%d)" % (att_delay, t, ctc_time))
                        ys.append(oldl)
                        tokens_to_update.append(c)
                        caches.append(hyps_att[oldl]['cache'])
                    else:
                        # logging.debug("Attention: But not augmenting due to insufficient context!")
                        pass
                else:
                    oldoldl = l[:-2]
                    c = l[-2]
                    assert oldoldl in hyps_att, "att hyp lagging by more than TWO step!!!"
                    att_time = hyps_att[oldoldl]['att_times'][-1]
                    logging.debug("\nAttention: Hyp lagging by two steps!")
                    logging.debug("Attention: Will add %s to [%s], in order to catch [%s]" %
                          (char_list[c], " ".join([char_list[_] for _ in oldoldl]), " ".join([char_list[_] for _ in l]) ))
                    logging.debug("Attention: current t=%d, att_time=%d, ctc_time=%d, while att_delay=%d" %
                          (t, att_time, hyps_ctc[l]['ctc_times'][-1], att_delay))
                    ys.append(oldoldl)
                    tokens_to_update.append(c)
                    caches.append(hyps_att[oldoldl]['cache'])

        # Weiran: one thing we could do to reduce computation is to collect those unique ys.
        logging.debug("\n")
        logging.debug("#" * 40)
        num_to_update = len(ys)
        logging.debug("Forwarding %d attention hyps ..." % num_to_update)
        if num_to_update > 0:
            # Memory up to frame t.
            hs_pad = enc_output[:, :min(t+1+dec_delay, T), :].repeat([num_to_update, 1, 1])
            local_att_scores, new_caches = model.decoder.batch_forward_one_step_with_cache(ys, caches, hs_pad, None)
            local_att_scores = local_att_scores.numpy()

            for i in range(num_to_update):
                oldl = ys[i]
                newl = ys[i] + (tokens_to_update[i],)
                newdict = dict()
                logging.debug("Attention: Appending %s to [%s], add to att score %s" %
                        (char_list[tokens_to_update[i]], " ".join([char_list[_] for _ in oldl]),
                        local_att_scores[i, tokens_to_update[i]]))
                newdict['att_score'] = hyps_att[oldl]['att_score'] + local_att_scores[i, tokens_to_update[i]]
                newdict['att_score_last'] = local_att_scores[i, tokens_to_update[i]]
                newdict['att_times'] = hyps_att[oldl]['att_times'] + [min(t+dec_delay, T-1)]
                newdict['cache'] = new_caches[i]
                hyps_att[newl] = newdict
                # If some ctc new hyps can be premature, the actual time delay can be too short??
            logging.debug("Computing attention scores finished ...")
            del new_caches
        logging.debug("\n")

        # Obtain joint score.
        if not t==T-1:
            logging.debug(">" * 40)
            joint_hyps = []
            for l in hyps_ctc:
                total_score = recog_args.lm_weight * hyps_ctc[l]['accum_lmscore'] + penalty * (len(l) - 1)
                ctc_score = logsumexp(*hyps_ctc[l]['ctc_scores'])
                # Weiran: my implementation of combined score.
                if l in hyps_att:
                    att_score = hyps_att[l]['att_score']
                    total_score += ctc_weight * ctc_score + (1 - ctc_weight) * att_score
                else:
                    att_score = hyps_att[l[:-1]]['att_score']
                    total_score += ctc_weight * hyps_ctc[l]['prev_score'] + (1 - ctc_weight) * att_score + (ctc_score - hyps_ctc[l]['prev_score'])
                """
                # The choice in MERL paper.
                if l in hyps_att:
                    att_score = hyps_att[l]['att_score']
                else:
                    att_score = hyps_att[l[:-1]]['att_score']
                total_score += ctc_weight * ctc_score + (1 - ctc_weight) * att_score
                """
                joint_hyps.append((l, total_score))

            joint_hyps = dict(sorted(joint_hyps, key=lambda x: x[1], reverse=True)[:beam_size])
            logging.debug("JOINT hyps ...")
            for l in list(joint_hyps.keys())[:min(10, len(joint_hyps))]:
                logging.debug("hyp: [%s], total score: %f" % ("".join([char_list[x] for x in l]), joint_hyps[l]))

            # Prune again the CTC hyps.
            final_ctc_keys = ctc_top_keys.union(set(joint_hyps.keys()))
            del joint_hyps
            # Clean dictionaries.
            for l in list(hyps_ctc.keys()):
                if l not in final_ctc_keys:
                    # logging.debug("Removing hypo [%s] from hyps_ctc" % " ".join([char_list[_] for _ in l]))
                    hyps_ctc.pop(l)
            final_att_keys = set([l[:-1] for l in final_ctc_keys]).union(final_ctc_keys)
            for l in list(hyps_att.keys()):
                if l not in final_att_keys:
                    # logging.debug("Removing hypo [%s] from hyps_ctc" % " ".join([char_list[_] for _ in l]))
                    hyps_att.pop(l)
    # Finished loop over time.

    # FINAL STAGES.
    logging.debug("*" * 80)
    logging.debug("CTC finished all %d frames ..." % T)
    logging.debug("\n")
    logging.debug("Attention: Clear one-step delays ...")
    ys = []
    tokens_to_update = []
    caches = []
    t = T - 1
    for l in hyps_ctc:
        if l not in hyps_att:
            oldl = l[:-1]
            assert oldl in hyps_att, "att can not be lagging by more than ONE step!!!"
            c = l[-1]
            ctc_time = hyps_ctc[l]['ctc_times'][-1]
            # Even if there is not sufficient number of frames since CTC added the token c.
            logging.debug("Attention: Will append %s to [%s]" %
                  (char_list[c], " ".join([char_list[_] for _ in oldl])))
            logging.debug("Attention: Actual delay %d (current t=%d, ctc_time=%d)" % (t-ctc_time, t, ctc_time))
            ys.append(oldl)
            tokens_to_update.append(c)
            caches.append(hyps_att[oldl]['cache'])

    # Weiran: one thing we could do to reduce computation is to collect those unique ys.
    num_to_update = len(ys)
    logging.debug("Forwarding %d attention hyps ..." % num_to_update)
    if num_to_update > 0:
        hs_pad = enc_output[:, :(t + 1), :].repeat([num_to_update, 1, 1])

        local_att_scores, new_caches = model.decoder.batch_forward_one_step_with_cache(ys, caches, hs_pad, None)
        local_att_scores = local_att_scores.numpy()

        for i in range(num_to_update):
            oldl = ys[i]
            newl = ys[i] + (tokens_to_update[i],)
            newdict = dict()
            logging.debug("Attention: Appending %s to [%s], add to att score %f" %
                  (char_list[tokens_to_update[i]], " ".join([char_list[_] for _ in oldl]),
                   local_att_scores[i, tokens_to_update[i]]))
            newdict['att_score'] = hyps_att[oldl]['att_score'] + local_att_scores[i, tokens_to_update[i]]
            newdict['att_score_last'] = local_att_scores[i, tokens_to_update[i]]
            newdict['att_times'] = hyps_att[oldl]['att_times'] + [t]
            newdict['cache'] = new_caches[i]
            hyps_att[newl] = newdict
            # If some ctc new hyps can be premature, the actual time delay can be too short.
        del new_caches
        logging.debug("Computing attention scores finished ...")

    # Final final step.
    logging.debug("\nCTC rnnlm_scores update with <eos> ...")
    for l in hyps_ctc:
        hyps_ctc[l]['accum_lmscore'] = hyps_ctc[l]['accum_lmscore'] + hyps_ctc[l]['rnnlm_scores'][eos_idx]

    logging.debug("\nAttention adding <eos> to hyps ...")
    ys = []
    tokens_to_update = []
    caches = []
    for l in hyps_att:
        ys.append(l)
        tokens_to_update.append(eos_idx)
        caches.append(hyps_att[l]['cache'])

    # Weiran: one thing we could do to reduce computation is to collect those unique ys.
    num_to_update = len(ys)
    logging.debug("Forwarding %d attention hyps ..." % num_to_update)
    if num_to_update > 0:
        hs_pad = enc_output[:, :(t + 1), :].repeat([num_to_update, 1, 1])

        local_att_scores, new_caches = model.decoder.batch_forward_one_step_with_cache(ys, caches, hs_pad, None)
        local_att_scores = local_att_scores.numpy()

        for i in range(num_to_update):
            oldl = ys[i]
            logging.debug("Attention: Appending %s to [%s], add to att score %f" %
                  (char_list[tokens_to_update[i]], " ".join([char_list[_] for _ in oldl]),
                   local_att_scores[i, tokens_to_update[i]]))
            hyps_att[oldl]['att_score'] = hyps_att[oldl]['att_score'] + local_att_scores[i, tokens_to_update[i]]
        del new_caches
        logging.debug("Computing attention final scores finished ...")
    logging.debug("\n")

    joint_hyps = []
    for l in hyps_ctc:
        ctc_score = logsumexp(*hyps_ctc[l]['ctc_scores'])
        att_score = hyps_att[l]['att_score']
        total_score = ctc_weight * ctc_score + (1 - ctc_weight) * att_score + recog_args.lm_weight * hyps_ctc[l][
            'accum_lmscore'] + penalty * len(l)
        joint_hyps.append((l + (eos_idx,), total_score))
    joint_hyps = sorted(joint_hyps, key=lambda x: x[1], reverse=True)[:min(len(joint_hyps), recog_args.nbest)]

    # Get consistent output format.
    return [{'yseq': list(l), 'score': v, 'ctc_times': hyps_ctc[l[:-1]]['ctc_times'], 'att_times': hyps_att[l[:-1]]['att_times'] + [T-1]} for l, v in joint_hyps]
