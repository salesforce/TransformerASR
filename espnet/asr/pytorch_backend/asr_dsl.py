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

"""Training/decoding definition for the speech recognition task."""
import pdb
import json
import logging
import math
import os
import shutil
import tempfile
import time

import numpy as np
from tensorboardX import SummaryWriter
import torch

import torch.distributed as dist

from speech_datasets import SpeechDataLoader
from espnet.asr.pytorch_backend.asr_init import load_trained_modules
from espnet.nets.asr_interface import ASRInterface
from espnet.utils.deterministic_utils import set_deterministic_pytorch
from espnet.utils.dynamic_import import dynamic_import

from espnet.asr.pytorch_backend.pytorch_distributed_utils import DistributedModel
from espnet.asr.pytorch_backend.pytorch_distributed_utils import all_gather_list

from espnet.nets.pytorch_backend.nets_utils import pad_list
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.subsampling import _context_concat

# For recognizer.
from espnet.asr.pytorch_backend.asr_init import load_trained_model
import espnet.lm.pytorch_backend.extlm_bpe as extlm_pytorch
import espnet.nets.pytorch_backend.lm.default as lm_pytorch

from espnet.asr.asr_utils import get_model_conf
from espnet.asr.asr_utils import torch_load
from espnet.asr.asr_utils import add_results_to_json
from espnet.asr.asr_utils import add_results_to_json_word

CTC_SCORING_RATIO=1.5

def _recursive_to(xs, device):
    if torch.is_tensor(xs):
        return xs.to(device)
    if isinstance(xs, tuple):
        return tuple(_recursive_to(x, device) for x in xs)
    return xs


class CustomConverter(object):
    """Custom batch converter for Pytorch.

    Args:
        input_context (int): The context to be augmented to the left and right of each frame.
        input_skiprate (int): The subsampling factor.
        dtype (torch.dtype): Data type to convert.

    """

    def __init__(self, input_context=0, input_skiprate=1, mode="eval", dtype=torch.float32):
        """Construct a CustomConverter object."""
        self.input_context = input_context
        self.input_skiprate = input_skiprate
        self.ignore_id = -1
        self.mode = mode
        self.dtype = dtype

    def __call__(self, batch, device=torch.device('cpu')):
        """Transform a batch and send it to a device.

        Args:
            batch (list): The batch to transform.
            device (torch.device): The device to send to.

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor)

        """
        # batch should be a list
        xs = [_['x'] for _ in batch]
        ys = [_['labels'] for _ in batch]

        # perform subsampling
        if self.input_context > 0:
            xs = [_context_concat(x, self.input_context) for x in xs]
        if self.input_skiprate > 1:
            if self.mode == "train":
                startidx = np.random.randint(low=0, high=2*self.input_context+1)
            else:
                startidx = 0
            xs = [x[startidx::self.input_skiprate, :] for x in xs]

        # get batch of lengths of input sequences
        ilens = np.array([x.size(0) for x in xs])

        # perform padding and convert to tensor
        # currently only support real number
        xs_pad = pad_list([x.float() for x in xs], 0).to(device, dtype=self.dtype)

        ilens = torch.from_numpy(ilens).to(device)
        # NOTE: this is for multi-output (e.g., speech translation)
        ys_pad = pad_list([y.long() for y in ys], self.ignore_id).to(device)

        return xs_pad, ilens, ys_pad


def torch_resume(snapshot_path, model, optimizer):
    """Resume from snapshot for pytorch.

    Args:
        snapshot_path (str): Snapshot file path.
        model: ASR model object.

    """
    # load snapshot
    snapshot_dict = torch.load(snapshot_path, map_location=lambda storage, loc: storage)

    # restore model states
    # (for ASR model)
    if hasattr(model, "module"):
        model.module.load_state_dict(snapshot_dict['model'])
    else:
        model.load_state_dict(snapshot_dict['model'])

    # restore optimizer states
    optimizer.load_state_dict(snapshot_dict['optimizer'])

    train_dict = snapshot_dict['train_dict']
    # delete opened snapshot
    del snapshot_dict
    return train_dict


def torch_snapshot(model, optimizer, train_dict, fn, outdir):
    # make snapshot_dict dictionary
    # (for ASR)
    if hasattr(model, "module"):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    snapshot_dict = {
        "model": model_state_dict,
        "optimizer": optimizer.state_dict(),
        "train_dict": train_dict
    }

    # save snapshot dictionary
    prefix = 'tmp' + fn
    tmpdir = tempfile.mkdtemp(prefix=prefix, dir=outdir)
    tmppath = os.path.join(tmpdir, fn)
    try:
        torch.save(snapshot_dict, tmppath)
        shutil.move(tmppath, os.path.join(outdir, fn))
    finally:
        shutil.rmtree(tmpdir)


# Multiply the gradient by a scalar.
def multiply_grads(model, c):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.mul_(c)


def train(args):
    """Train with the given args.

    Args:
        args (namespace): The program arguments.

    """

    # Get the rank of current process. Determines the portion of data to load.
    global_rank = dist.get_rank()

    set_deterministic_pytorch(args)

    # check cuda availability
    if not torch.cuda.is_available():
        logging.error('cuda is not available')

    # idim and odim are specified by user.
    idim = args.idim
    odim = args.odim
    if global_rank == 0:
        logging.info('#input dims : {}'.format(idim))
        logging.info('#output dims: {}' .format(odim))

    # specify attention, CTC, hybrid mode
    if global_rank == 0:
        if args.mtlalpha == 1.0:
            logging.info('Pure CTC mode')
        elif args.mtlalpha == 0.0:
            logging.info('Pure attention mode')
        else:
            logging.info('Multitask learning mode')

    if (args.enc_init is not None or args.dec_init is not None):
        model = load_trained_modules(idim * (2 * args.input_context + 1), odim, args)
    else:
        model_class = dynamic_import(args.model_module)
        model = model_class(idim * (2 * args.input_context + 1), odim, args)
    assert isinstance(model, ASRInterface)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    logging.info("total number of params in model: %d" % params)

    # write model config
    if global_rank == 0 and not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    dist.barrier()

    model_conf = args.outdir + '/model.json'
    if global_rank == 0:
        with open(model_conf, 'wb') as f:
            logging.info('writing a model config file to ' + model_conf)
            f.write(json.dumps((idim * (2 * args.input_context + 1), odim, vars(args)),
                               indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))
        for key in sorted(vars(args).keys()):
            logging.info('ARGS: ' + key + ': ' + str(vars(args)[key]))
        # Reporter.
        reporter = SummaryWriter(args.tensorboard_dir)

    # check the use of multi-gpu
    world_size = dist.get_world_size()
    assert args.ngpu > 0 and world_size > 0, "Distributed training requires GPUs ..."
    if global_rank == 0:
        logging.warning(
            'batch size is automatically increased (%d -> %d)' % (args.batch_size, args.batch_size * world_size))

    # Set torch device
    torch.cuda.set_device(args.local_rank)
    model = model.cuda(args.local_rank)
    if args.train_dtype in ("float16", "float32", "float64"):
        dtype = getattr(torch, args.train_dtype)
    else:
        dtype = torch.float32
    device = args.local_rank

    # Setup an optimizer
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.adam_lr, weight_decay=args.weight_decay)
    elif args.opt == 'noam':
        from espnet.nets.pytorch_backend.transformer.optimizer import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.transformer_warmup_steps, args.transformer_lr)
    elif args.opt == 'rampup':
        from espnet.nets.pytorch_backend.transformer.rampup import get_std_opt
        optimizer = get_std_opt(model, args.adim, args.rampup_sr, args.rampup_si, args.rampup_sf, args.rampup_lr)
    else:
        raise NotImplementedError("unknown optimizer: " + args.opt)

    # Resume from a snapshot
    if args.resume:
        train_dict = torch_resume(args.resume, model, optimizer)
        if global_rank == 0:
            logging.info('resumed from %s' % args.resume)
    else:
        train_dict = dict({"epoch": 0, "iteration": 0, "validation_loss": []})

    model = DistributedModel(model, args.local_rank, bucket_cap_mb=256)

    # Setup a converter
    train_converter = CustomConverter(input_context=args.input_context, input_skiprate=args.input_skiprate,
                                      mode="train", dtype=dtype)
    valid_converter = CustomConverter(input_context=args.input_context, input_skiprate=args.input_skiprate,
                                      mode="eval", dtype=dtype)

    # create data loaders. train_sets and valid_sets will be dataset names separated by comma
    train_data = list(filter(lambda s: len(s) > 0, map(lambda s: s.strip(), args.train_sets.split(","))))
    valid_data = list(filter(lambda s: len(s) > 0, map(lambda s: s.strip(), args.valid_sets.split(","))))
    train_loader = SpeechDataLoader(train_data, task="asr", shuffle=True, precomputed_feats_type=args.precomputed_feats_type,
                          batch_size=args.batch_size, max_len=args.maxlen_in, spmodel=args.spmodel, token_list=args.char_list,
                          transform_conf=args.preprocess_conf, train=True, num_workers=args.loader_num_worker, data_cache_mb=args.loader_cache_mb,
                          text_filename=args.text_filename)
    valid_loader = SpeechDataLoader(valid_data, task="asr", shuffle=False, precomputed_feats_type=args.precomputed_feats_type,
                          batch_size=args.batch_size, max_len=args.maxlen_in, spmodel=args.spmodel, token_list=args.char_list,
                          transform_conf=args.preprocess_conf, train=False, num_workers=args.loader_num_worker, data_cache_mb=args.loader_cache_mb,
                          text_filename=args.text_filename)

    epoch = train_dict['epoch']
    train_loader.set_epoch(epoch)

    # These variables may be nonzero after resuming from snapshot.
    # Record the number of updates.
    iteration = train_dict["iteration"]
    # Record the validation loss.
    validation_loss = train_dict["validation_loss"]

    # Determine whether to update, and the accurate number of samples used between two updates.
    forward_count = 0
    xnum_train = 0
    ynum_train = 0
    loss_train = 0.0
    loss_att_train = 0.0
    loss_ctc_train = 0.0

    acc_train = 0.0

    while True:

        if 0 < args.epochs <= epoch:
            train_loader.close()
            valid_loader.close()
            break

        with model.no_sync():
            # Only synchronize the gradients every accum_grad steps, or at the end of an epoch
            while forward_count < args.accum_grad - 1 and train_loader.current_position < len(train_loader)-1:

                # Get the next batch
                logging.info("Getting batch from dataloader")
                batch = train_loader.next()
                logging.info("CHECK I: position=%d, total=%d, epoch=%d, model_device=%s" % (train_loader.current_position, len(train_loader), train_loader.epoch, next(model.parameters()).device))
                forward_count += 1

                x = _recursive_to(train_converter(batch), device)
                logging.info("Move batch to GPU")

                # Weiran: the actual number of utts in the minibatch.
                x_num = x[0].size(0)
                y_num = float(torch.sum(x[2] != train_converter.ignore_id).cpu())

                loss, loss_data, loss_ctc_data, loss_att_data, acc_data, _ = model(*x)

                xnum_train += x_num
                ynum_train += y_num
                loss_train += x_num * loss_data
                loss_att_train += x_num * loss_att_data
                loss_ctc_train += x_num * loss_ctc_data
                acc_train += y_num * acc_data

                # loss is the sum, not average.
                loss = x_num * loss
                loss.backward()
                loss.detach()  # Truncate the graph

        # Perform the same loop as above, but sync the gradients if it's been
        # accum_grad steps, or the epoch is about to finish
        logging.info("Getting batch from dataloader")
        batch = train_loader.next()
        logging.info("CHECK II: position=%d, total=%d, epoch=%d, model_device=%s" % (train_loader.current_position, len(train_loader), train_loader.epoch, next(model.parameters()).device))
        forward_count += 1

        x = _recursive_to(train_converter(batch), device)
        logging.info("Move batch to GPU")

        # Weiran: the actual number of utts in the minibatch.
        x_num = x[0].size(0)
        y_num = float(torch.sum(x[2] != train_converter.ignore_id).cpu())

        loss, loss_data, loss_ctc_data, loss_att_data, acc_data, _ = model(*x)

        xnum_train += x_num
        ynum_train += y_num
        loss_train += x_num * loss_data
        loss_att_train += x_num * loss_att_data
        loss_ctc_train += x_num * loss_ctc_data
        acc_train += y_num * acc_data

        # loss is the sum, not average.
        loss = x_num * loss
        loss.backward()
        loss.detach()  # Truncate the graph

        # update parameters
        is_new_epoch = (not epoch == train_loader.epoch)
        assert is_new_epoch or forward_count == args.accum_grad

        # Needed when distributed_world_size > 1.
        xnum_all, ynum_all, loss_all, loss_att_all, loss_ctc_all, acc_all = \
            zip(*all_gather_list([xnum_train, ynum_train, loss_train, loss_att_train, loss_ctc_train, acc_train]))
        total_xnum = sum(xnum_all)
        total_ynum = sum(ynum_all)
        total_loss = sum(loss_all)
        total_loss_att = sum(loss_att_all)
        total_loss_ctc = sum(loss_ctc_all)
        total_acc = sum(acc_all)

        # Re-scale gradients, in order to obtain the right loss gradient with simple averaging.
        grad_factor = 1.0 / total_xnum
        multiply_grads(model, grad_factor)

        # compute the gradient norm to check if it is normal or not
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        if global_rank == 0:
            logging.info('grad norm={}'.format(grad_norm))
        if math.isnan(grad_norm):
            logging.warning('grad norm is nan. Do not update model.')
        else:
            optimizer.step()
        optimizer.zero_grad()

        iteration += 1
        if iteration % args.report_interval_iters == 0 and global_rank == 0:
            reporter.add_scalar("train/loss", total_loss / total_xnum, iteration)
            reporter.add_scalar("train/loss_att", total_loss_att / total_xnum, iteration)
            reporter.add_scalar("train/loss_ctc", total_loss_ctc / total_xnum, iteration)
            reporter.add_scalar("train/acc", total_acc / total_ynum, iteration)
            reporter.add_scalar("train/lr", optimizer.param_groups[0]['lr'], iteration)

        # Reset the accumulation.
        forward_count = 0
        xnum_train = 0
        ynum_train = 0
        loss_train = 0.0
        loss_att_train = 0.0
        loss_ctc_train = 0.0
        acc_train = 0.0

        # Finished one epoch.
        if is_new_epoch:
            epoch += 1

            # Evaluate the model with the validation dataset at the end of every epoch.
            model.eval()
            logging.info("Start validation for epoch %d" % train_loader.epoch)

            xnum_valid = 0
            ynum_valid = 0
            loss_valid = 0.0
            loss_att_valid = 0.0
            loss_ctc_valid = 0.0
            err_ctc_valid = 0.0
            acc_valid = 0.0

            for batch in valid_loader:

                logging.info("Getting batch from dataloader")
                x = _recursive_to(valid_converter(batch), device)
                logging.info("Move batch to GPU")

                x_num = x[0].size(0)
                y_num = float(torch.sum(x[2] != valid_converter.ignore_id).cpu())

                with torch.no_grad():
                    _, loss_data, loss_ctc_data, loss_att_data, acc_data, cer_ctc_data = model(*x)

                xnum_valid += x_num
                ynum_valid += y_num
                loss_valid += x_num * loss_data
                loss_att_valid += x_num * loss_att_data
                loss_ctc_valid += x_num * loss_ctc_data
                acc_valid += y_num * acc_data
                err_ctc_valid += y_num * cer_ctc_data

            xnum_all, ynum_all, loss_all, loss_att_all, loss_ctc_all, acc_all, err_ctc_all = \
                zip(*all_gather_list(
                    [xnum_valid, ynum_valid, loss_valid, loss_att_valid, loss_ctc_valid, acc_valid, err_ctc_valid]))

            total_xnum = sum(xnum_all)
            total_ynum = sum(ynum_all)
            total_loss = sum(loss_all)
            total_loss_att = sum(loss_att_all)
            total_loss_ctc = sum(loss_ctc_all)
            total_acc = sum(acc_all)
            total_err_ctc = sum(err_ctc_all)
            # Each GPU has access to the validation loss in order to adjust learning rate.
            validation_loss.append(total_loss / total_xnum)
            if global_rank == 0:
                reporter.add_scalar("valid/loss", total_loss / total_xnum, iteration)
                reporter.add_scalar("valid/loss_att", total_loss_att / total_xnum, iteration)
                reporter.add_scalar("valid/loss_ctc", total_loss_ctc / total_xnum, iteration)
                reporter.add_scalar("valid/acc", total_acc / total_ynum, iteration)
                reporter.add_scalar("valid/err_ctc", total_err_ctc / total_ynum, iteration)

            # Save model at the end of each epoch.
            if global_rank == 0:
                train_dict["epoch"] = train_loader.epoch
                train_dict["iteration"] = iteration
                train_dict["validation_loss"] = validation_loss
                torch_snapshot(model, optimizer, train_dict, 'snapshot.ep.%d' % train_loader.epoch, args.outdir)
                logging.info("snapshot saved to snapshot.ep.%d" % train_loader.epoch)
                if validation_loss[-1] == min(validation_loss):
                    torch_snapshot(model, optimizer, train_dict, 'model.loss.best', args.outdir)
                    logging.info("best model saved to model.loss.best")

            # Go back to training again.
            model.train()

            if args.opt == "adam":
                if epoch > 3 and min(validation_loss) < min(validation_loss[-3:]):
                    for p in optimizer.param_groups:
                        p["lr"] = max(p["lr"] * args.adam_decay, 1e-6)
                        if global_rank == 0:
                            logging.info('adam lr decayed to ' + str(p["lr"]))


def recog(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.
    """
    set_deterministic_pytorch(args)
    start_time=time.time()
    model, train_args = load_trained_model(args.model)
    end_time=time.time()
    logging.info("loading model took %f seconds" % (end_time-start_time))
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

    if args.word_rnnlm:
        rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
        word_dict = rnnlm_args.char_list_dict
        char_dict = {x: i for i, x in enumerate(train_args.char_list)}
        word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            max(word_dict.values()) + 1, rnnlm_args.layer, rnnlm_args.unit))
            # Weiran: modified the code to infer n_vocab when there are missing keys in char_list_dict.
            # len(word_dict), rnnlm_args.layer, rnnlm_args.unit))
        torch_load(args.word_rnnlm, word_rnnlm)
        word_rnnlm.eval()

        if rnnlm is not None:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.MultiLevelLM_with_lexicon(word_rnnlm.predictor,
                        rnnlm.predictor, word_dict, char_dict, args.lexicon_dict, "▁", subwordlm_weight=args.sublm_weight))
        else:
            rnnlm = lm_pytorch.ClassifierWithState(
                extlm_pytorch.LookAheadWordLM(word_rnnlm.predictor, word_dict, char_dict))

    # Read truth file.
    if args.truth_file:
        with open(args.truth_file, "r") as fin:
            retval = fin.read().rstrip('\n')
            dict_truth = dict([(l.split()[0], " ".join(l.split()[1:])) for l in retval.split("\n")])

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model.cuda()
        if rnnlm:
            rnnlm.cuda()

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
            # pdb.set_trace()
            name = batch[0]['uttid']
            logging.info('(%d/%d) decoding ' + name, idx, len(recog_loader))
            feat = _recursive_to(recog_converter(batch), device=torch.device('cpu'))[0]
            feat = feat[0]

            inverse_subword_dict = dict([(i,c) for i,c in enumerate(train_args.char_list)])
            if args.word_rnnlm:
                inverse_word_dict = dict([(word_dict[k], k) for k in word_dict])
            else:
                inverse_word_dict = None
            if args.rnnlm and args.word_rnnlm:
                nbest_hyps = recognize_with_lexicon(model, feat, args, rnnlm=rnnlm, inverse_subword_dict=inverse_subword_dict, inverse_word_dict=inverse_word_dict)
            else:
                nbest_hyps = model.recognize(feat, args, train_args.char_list, rnnlm)

            if args.truth_file and name in dict_truth:
                truth_text = dict_truth[name]
            else:
                truth_text = ""

            # Weiran: prepare dict in order to add decoding results. Skipped the input and shape information.
            gt_tokens = [int(_) for _ in batch[0]["labels"]]
            tmp_dict = {"output": [{"name": "target1", "text": batch[0]["text"],
                                    "tokenid": " ".join([str(_) for _ in gt_tokens]).strip(),
                                    "token": " ".join([train_args.char_list[_] for _ in gt_tokens]).strip()}],
                                    "utt2spk": batch[0]["speaker"]}

            # Weiran: I am adding text in words in the result json.
            if args.word_rnnlm and args.rnnlm:
                new_js[name] = add_results_to_json_word(tmp_dict, nbest_hyps, train_args.char_list, inverse_word_dict, truth_text)
            else:
                new_js[name] = add_results_to_json(tmp_dict, nbest_hyps, train_args.char_list, add_hyp_prefix_wer=args.nbest_compute_wer)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))



################################################################################################
################## Weiran: The recognizers below are implemented by myself. ####################
################################################################################################

def recognize_with_lexicon(model, x, recog_args, rnnlm=None, inverse_subword_dict=None, inverse_word_dict=None):
    """Recognize input speech.

    :model the asr model, such as e2e_asr_transformer:E2E
    :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
    :param Namespace recog_args: argument Namespace containing options
    :param torch.nn.Module rnnlm: language model module
    :param dict inverse_subword_dict: index to str
    :param dict inverse_word_dict: index to str
    :return: N-best decoding results
    :rtype: list
    """

    incomplete_word = -99

    enc_output = model.encode(x).unsqueeze(0)
    if recog_args.ctc_weight > 0.0:
        lpz = model.ctc.log_softmax(enc_output)
        lpz = lpz.squeeze(0)
    else:
        lpz = None

    h = enc_output.squeeze(0)

    logging.info('input lengths: ' + str(h.size(0)))
    # search parms
    beam = recog_args.beam_size
    ctc_weight = recog_args.ctc_weight
    word_bonus = recog_args.word_bonus

    # prepare sos
    y = model.sos
    vy = h.new_zeros(1).long()

    if recog_args.maxlenratio == 0:
        maxlen = h.shape[0]
    else:
        # maxlen >= 1
        maxlen = max(1, int(recog_args.maxlenratio * h.size(0)))
    minlen = int(recog_args.minlenratio * h.size(0))
    logging.info('max output length: ' + str(maxlen))
    logging.info('min output length: ' + str(minlen))

    # initialize hypothesis
    if rnnlm:
        hyp = {'score': 0.0, 'yseq': [y], 'wseq': [], 'rnnlm_prev': None}
    else:
        hyp = {'score': 0.0, 'yseq': [y], 'wseq': []}

    if lpz is not None:
        import numpy
        from espnet.nets.ctc_prefix_score import CTCPrefixScore

        ctc_prefix_score = CTCPrefixScore(lpz.detach().numpy(), 0, model.eos, numpy)
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
    for i in six.moves.range(maxlen):
        logging.debug('position ' + str(i))

        hyps_best_kept = []
        for hyp in hyps:
            # vy.unsqueeze(1)
            # Weiran: get the last hypothesized token.
            vy[0] = hyp['yseq'][i]

            # get nbest local scores and their ids
            ys_mask = subsequent_mask(i + 1).unsqueeze(0)
            ys = torch.tensor(hyp['yseq']).unsqueeze(0)
            local_att_scores = model.decoder.forward_one_step(ys, ys_mask, enc_output)[0]

            if rnnlm:
                logging.debug("\nUsing rnnlm ...")
                RNNLM_STATE_LIST = rnnlm.predict(hyp['rnnlm_prev'], vy)
            else:
                # Fake the RNNLM list, with only one item.
                RNNLM_STATE_LIST = [(None, local_att_scores, incomplete_word)]

            # Weiran: if the list has more than one element, we need to expand the set of hypothesis.
            for rnnlm_state, local_lm_scores, word_output in RNNLM_STATE_LIST:

                if word_output>0:
                    logging.debug("\n===================\nHypothesis:")
                    logging.debug(hyp['yseq'])
                    logging.debug(" ".join([inverse_subword_dict[int(tid)] for tid in hyp['yseq']]))
                    logging.debug("current word hypothesis:")
                    logging.debug(hyp['wseq'])
                    logging.debug(" ".join([inverse_word_dict[int(wid)] for wid in hyp['wseq'] if wid > 0]))
                    if word_output>=0:
                        logging.debug("outputing word: %s (%d)" % (inverse_word_dict[int(word_output)], int(word_output)))
                    else:
                        logging.debug("Not outputing word.")
                    logging.debug("current score=%f" % hyp['score'])
                    logging.debug("acc. clm score=%f" % rnnlm_state[-1])

                if rnnlm:
                    local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores
                else:
                    local_scores = local_att_scores

                # Weiran: correct local_scores if ctc_prefix_score is used.
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

                # Going over the beams.
                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(local_best_ids[0, j])
                    new_hyp['wseq'] = list(hyp['wseq']) + [word_output]
                    if word_output >= 0:
                        new_hyp['score'] = new_hyp['score'] + word_bonus

                    if rnnlm:
                        new_hyp['rnnlm_prev'] = rnnlm_state

                    if lpz is not None:
                        new_hyp['ctc_state_prev'] = ctc_states[joint_best_ids[0, j]]
                        new_hyp['ctc_score_prev'] = ctc_scores[joint_best_ids[0, j]]

                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

        # Finish expanding all hypothesis at position i.
        # sort and get nbest
        hyps = hyps_best_kept
        logging.debug('number of pruned hypothes: ' + str(len(hyps)))
        if inverse_subword_dict is not None:
            logging.debug('best hypo: ' + ''.join([inverse_subword_dict[int(x)] for x in hyps[0]['yseq'][1:]]))

        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info('adding <eos> in the last position in the loop')
            for hyp in hyps:
                hyp['yseq'].append(model.eos)

        # add ended hypothesis to a final list, and removed them from current hypothes
        # (this will be a problem, number of hyps < beam)
        remained_hyps = []
        for hyp in hyps:
            if hyp['yseq'][-1] == model.eos:
                # only store the sequence that has more than minlen outputs
                if len(hyp['yseq']) > minlen:
                    if rnnlm:  # Word LM needs to add final <eos> score
                        RNNLM_FINAL_LIST = rnnlm.final(hyp['rnnlm_prev'])
                        for rnnlm_final_score, word_output in RNNLM_FINAL_LIST:

                            logging.debug("\n===================\nHypothesis ending:")
                            logging.debug(hyp['yseq'])
                            logging.debug(" ".join([inverse_subword_dict[int(tid)] for tid in hyp['yseq']]))
                            logging.debug("current word hypothesis:")
                            logging.debug(hyp['wseq'])
                            logging.debug(" ".join([inverse_word_dict[int(wid)] for wid in hyp['wseq'] if wid > 0]))
                            if word_output >= 0:
                                logging.debug("outputing word: %s" % inverse_word_dict[int(word_output)])
                            logging.debug("current score=%f" % hyp['score'])
                            logging.debug("adding last word+end score=%f\n===================" % (recog_args.lm_weight * rnnlm_final_score))

                            new_hyp = {}
                            new_hyp['score'] = hyp['score'] + recog_args.lm_weight * rnnlm_final_score
                            new_hyp['yseq'] = hyp['yseq']
                            new_hyp['wseq'] = list(hyp['wseq']) + [word_output]
                            if word_output >= 0:
                                new_hyp['score'] = new_hyp['score'] + word_bonus
                            ended_hyps.append(new_hyp)
            else:
                remained_hyps.append(hyp)

        # end detection
        from espnet.nets.e2e_asr_common import end_detect
        if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
            logging.info('end detected at %d' % i)
            break

        hyps = remained_hyps
        if len(hyps) > 0:
            logging.debug('remained hypothes: ' + str(len(hyps)))
        else:
            logging.info('no hypothesis. Finish decoding.')
            break

        if inverse_subword_dict is not None:
            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([inverse_subword_dict[int(x)] for x in hyp['yseq'][1:]]))

        logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))

    # Finished label-synchronous decoding.
    nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

    # check number of hypothesis
    if len(nbest_hyps) == 0:
        logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
        # should copy because Namespace will be overwritten globally
        from argparse import Namespace
        recog_args = Namespace(**vars(recog_args))
        recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
        return recognize_with_lexicon(model, x, recog_args, rnnlm=rnnlm, inverse_subword_dict=inverse_subword_dict, inverse_word_dict=inverse_word_dict)

    logging.info('total log probability: ' + str(nbest_hyps[0]['score']))
    logging.info('normalized log probability: ' + str(nbest_hyps[0]['score'] / len(nbest_hyps[0]['yseq'])))
    for hidx in range(min(10, len(nbest_hyps))):
        logging.debug("HYP%02d (score=%f): %s" % (hidx + 1, nbest_hyps[hidx]['score'], " ".join([inverse_word_dict[int(wid)] for wid in nbest_hyps[hidx]['wseq'] if wid > 0])))

    return nbest_hyps


def recog_with_two_models(args):
    """Decode with the given args.

    Args:
        args (namespace): The program arguments.
    """
    set_deterministic_pytorch(args)

    # Weiran: load both models.
    model1, train_args1 = load_trained_model(args.model1)
    assert isinstance(model1, ASRInterface)
    model1.recog_args = args

    # Read the rnnlm for model1.
    rnnlm1_args = get_model_conf(args.rnnlm1, args.rnnlm1_conf)
    subword_dict1 = rnnlm1_args.char_list_dict
    rnnlm1 = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(subword_dict1), rnnlm1_args.layer, rnnlm1_args.unit))
    torch_load(args.rnnlm1, rnnlm1)
    rnnlm1.eval()

    model2, train_args2 = load_trained_model(args.model2)
    assert isinstance(model2, ASRInterface)
    model2.recog_args = args

    # Read the rnnlm for model2.
    rnnlm2_args = get_model_conf(args.rnnlm2, args.rnnlm2_conf)
    subword_dict2 = rnnlm2_args.char_list_dict
    rnnlm2 = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
            len(subword_dict2), rnnlm2_args.layer, rnnlm2_args.unit))
    torch_load(args.rnnlm2, rnnlm2)
    rnnlm2.eval()

    # Weiran: There is only one word-level language model.
    word_rnnlm_args = get_model_conf(args.word_rnnlm, args.word_rnnlm_conf)
    word_dict = word_rnnlm_args.char_list_dict
    word_rnnlm = lm_pytorch.ClassifierWithState(lm_pytorch.RNNLM(
        max(word_dict.values()) + 1, word_rnnlm_args.layer, word_rnnlm_args.unit))
    torch_load(args.word_rnnlm, word_rnnlm)
    word_rnnlm.eval()

    # There two lexicons for the two models.
    rnnlm1 = lm_pytorch.ClassifierWithState(extlm_pytorch.MultiLevelLM_with_lexicon(word_rnnlm.predictor,
                    rnnlm1.predictor, word_dict, subword_dict1, args.lexicon1_dict, "▁", subwordlm_weight=args.sublm_weight))

    # Char-based system. There is no ambiguities in lexicon2_dict.
    with open(args.lexicon2_dict, "r") as fin:
        lines = fin.read().rstrip('\n').split("\n")
        lexicon2_dict = dict([(l.split()[0], l.split()[1:]) for l in lines])
    # Weiran (04/17/2020): switching to not using subwordLM from subword LM for model2.
    rnnlm2 = lm_pytorch.ClassifierWithState(extlm_pytorch.MultiLevelLM_with_lexicon(word_rnnlm.predictor,
                    rnnlm2.predictor, word_dict, subword_dict2, args.lexicon2_dict, "▁", subwordlm_weight=0.0))

    # Read truth file.
    if args.truth_file:
        with open(args.truth_file, "r") as fin:
            retval = fin.read().rstrip('\n')
            dict_truth = dict([(l.split()[0], " ".join(l.split()[1:])) for l in retval.split("\n")])

    # gpu
    if args.ngpu == 1:
        gpu_id = list(range(args.ngpu))
        logging.info('gpu id: ' + str(gpu_id))
        model1.cuda()
        model2.cuda()
        rnnlm1.cuda()
        rnnlm2.cuda()


    new_js = {}
    recog_converter = CustomConverter(input_context=args.input_context, input_skiprate=args.input_skiprate,
                                      mode="eval", dtype=torch.float32)
    recog_data = list(filter(lambda s: len(s) > 0, map(lambda s: s.strip(), args.recog_sets.split(","))))
    recog_loader = SpeechDataLoader(recog_data, task="asr", shuffle=False,
                                    precomputed_feats_type=args.precomputed_feats_type,
                                    batch_size=1, spmodel=args.spmodel, token_list=train_args1.char_list,
                                    transform_conf=args.preprocess_conf, train=False, num_workers=args.loader_num_worker,
                                    data_cache_mb=args.loader_cache_mb, num_replicas=args.num_replicas, rank=args.jobid-1,
                                    ensure_equal_parts=False, text_filename=args.text_filename)

    inverse_subword_dict1 = dict([(subword_dict1[k], k) for k in subword_dict1])
    inverse_subword_dict2 = dict([(subword_dict2[k], k) for k in subword_dict2])
    inverse_word_dict = dict([(word_dict[k], k) for k in word_dict])

    with torch.no_grad():
        idx=0
        for batch in recog_loader:
            idx += 1
            name = batch[0]['uttid']
            logging.info('(%d/%d) decoding ' + name, idx, len(recog_loader))
            feat = _recursive_to(recog_converter(batch), device=torch.device('cpu'))[0]
            feat = feat[0]

            nbest_hyps = recognize_with_two_transformers(feat, args, model1, rnnlm1, inverse_subword_dict1,
                                    model2, rnnlm2, lexicon2_dict, inverse_subword_dict2, inverse_word_dict)

            # pdb.set_trace()
            if args.truth_file and name in dict_truth:
                truth_text = dict_truth[name]
            else:
                truth_text = ""

            logging.info("groundtruth: %s" % truth_text)
            logging.info("Best hypothesis combined:")
            for hidx in range(min(10, len(nbest_hyps))):
                logging.info("HYP%02d (score=%f, score1=%f, score2=%f): %s" % (hidx+1, nbest_hyps[hidx]['score'], nbest_hyps[hidx]['score1'], nbest_hyps[hidx]['score2'],
                        " ".join([inverse_word_dict[int(wid)] for wid in nbest_hyps[hidx]['wseq'] if wid > 0])))

            # Weiran: prepare dict in order to add decoding results.
            # I skipped the input and shape information.
            gt_tokens = [int(_) for _ in batch[0]["labels"]]
            tmp_dict = {"output": [{"name": "target1", "text": batch[0]["text"],
                                    "tokenid": " ".join([str(_) for _ in gt_tokens]).strip(),
                                    "token": " ".join([train_args1.char_list[_] for _ in gt_tokens]).strip()}],
                                    "utt2spk": batch[0]["speaker"]}

            # Weiran: I am adding text in words in the result json.
            new_js[name] = add_results_to_json_word(tmp_dict, nbest_hyps, train_args1.char_list, inverse_word_dict, truth_text)

    with open(args.result_label, 'wb') as f:
        f.write(json.dumps({'utts': new_js}, indent=4, ensure_ascii=False, sort_keys=True).encode('utf_8'))


# Weiran: model1 is phone-BPE based, model2 is char-BPE based.
def recognize_with_two_transformers(x, recog_args, model1, rnnlm1, inverse_subword_dict1, model2, rnnlm2, lexicon2, inverse_subword_dict2, inverse_word_dict):
    """Recognize input speech.

    :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
    :param Namespace recog_args: argment Namespace contraining options
    :param torch.nn.Module rnnlm: language model module
    :param dict inverse_subword_dict: index to str
    :param dict inverse_word_dict: index to str
    :return: N-best decoding results
    :rtype: list
    """

    # subword_dict1 = dict([(inverse_subword_dict1[k], k) for k in inverse_subword_dict1])
    subword_dict2 = dict([(inverse_subword_dict2[k], k) for k in inverse_subword_dict2])

    # Search parameters.
    beam = recog_args.beam_size
    word_bonus = recog_args.word_bonus
    ctc_weight = recog_args.ctc_weight
    logging.info("using word_bonus=%f" % word_bonus)
    logging.info("using ctc_weight=%f" % ctc_weight)
    from espnet.nets.ctc_prefix_score import CTCPrefixScore

    # Prepare encoder outputs.
    enc_output1 = model1.encode(x).unsqueeze(0)
    lpz1 = model1.ctc.log_softmax(enc_output1).squeeze(0)
    h1 = enc_output1.squeeze(0)
    logging.info('input lengths: ' + str(h1.size(0)))
    if ctc_weight != 1.0:
        ctc1_beam = min(lpz1.shape[-1], int(beam * CTC_SCORING_RATIO))
    else:
        ctc1_beam = lpz1.shape[-1]

    enc_output2 = model2.encode(x).unsqueeze(0)
    lpz2 = model2.ctc.log_softmax(enc_output2).squeeze(0)
    h2 = enc_output2.squeeze(0)

    # Output length.
    if recog_args.maxlenratio == 0:
        maxlen = h1.shape[0]
    else:
        maxlen = max(1, int(recog_args.maxlenratio * h1.size(0)))
    minlen = int(recog_args.minlenratio * h1.size(0))
    logging.info('max output length: ' + str(maxlen))
    logging.info('min output length: ' + str(minlen))

    # Weiran: prepare sos. y1 is for token index, vy1 is tensor for rnnlm model.
    y1 = model1.sos
    vy1 = h1.new_zeros(1).long()
    y2 = model2.sos
    vy2 = h2.new_zeros(1).long()

    # initialize hypothesis
    hyp = {'score': 0.0, 'score1': 0.0, 'yseq': [y1], 'wseq': [], 'rnnlm1_prev': None,
                         'score2': 0.0, 'yseq2': [y2], 'wseq2': [], 'rnnlm2_prev': None}

    # CTC scoring.
    ctc1_prefix_score = CTCPrefixScore(lpz1.detach().numpy(), 0, model1.eos, np)
    hyp['ctc1_state_prev'] = ctc1_prefix_score.initial_state()
    hyp['ctc1_score_prev'] = 0.0

    ctc2_prefix_score = CTCPrefixScore(lpz2.detach().numpy(), 0, model2.eos, np)
    hyp['ctc2_state_prev'] = ctc2_prefix_score.initial_state()
    hyp['ctc2_score_prev'] = 0.0

    # Main loop.
    hyps = [hyp]
    ended_hyps = []

    # Weiran: word-synchronous decoding.
    import six
    for i in six.moves.range(maxlen):
        logging.debug('position ' + str(i))

        hyps_best_kept = []
        for hyp in hyps:

            logging.debug("\n&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&\n")
            logging.debug("Extending hypothesis %s" % (" ".join([inverse_subword_dict1[int(tid)] for tid in hyp['yseq']])))
            # Weiran: use model1 to propose new word hypothesis.
            # vy1.unsqueeze(1)
            # Weiran: get the last hypothesized token.
            vy1[0] = hyp['yseq'][i]

            # get nbest local scores and their ids
            ys_mask1 = subsequent_mask(i + 1).unsqueeze(0)
            ys1 = torch.tensor(hyp['yseq']).unsqueeze(0)
            local_att_scores = model1.decoder.forward_one_step(ys1, ys_mask1, enc_output1)[0]

            logging.debug("\nUsing rnnlm1 ...")
            RNNLM_STATE_LIST = rnnlm1.predict(hyp['rnnlm1_prev'], vy1)

            # Weiran: if the list has more than one element, we need to expand the set of hypothesis.
            for rnnlm1_state, local_lm_scores1, word_output1 in RNNLM_STATE_LIST:

                logging.debug("\nFor rnnlm1 word_output1=%d" % word_output1)

                local_scores = local_att_scores + recog_args.lm_weight * local_lm_scores1
                local_best_scores, local_best_ids = torch.topk(local_scores, ctc1_beam, dim=1)
                ctc_scores, ctc_states = ctc1_prefix_score(hyp['yseq'], local_best_ids[0], hyp['ctc1_state_prev'])
                local_scores = (1.0 - ctc_weight) * local_att_scores[:, local_best_ids[0]] \
                               + ctc_weight * torch.from_numpy(ctc_scores - hyp['ctc1_score_prev']) \
                               + recog_args.lm_weight * local_lm_scores1[:, local_best_ids[0]]
                local_best_scores, joint_best_ids = torch.topk(local_scores, beam, dim=1)
                local_best_ids = local_best_ids[:, joint_best_ids[0]]

                # Prepare for model2, since it may go through a loop when the word boundary is met by model1.
                yseq2 = list(hyp['yseq2'])
                wseq2 = list(hyp['wseq2'])
                score2 = hyp['score2']
                ctc2_state_prev = hyp['ctc2_state_prev']
                ctc2_score_prev = hyp['ctc2_score_prev']
                rnnlm2_state = hyp['rnnlm2_prev']

                if word_output1 >= 0:
                    if inverse_word_dict[word_output1] == "<unk>":
                        # Weiran (05/08/2020): shall consider changing this to <unk> of model2, whose ID is 1.
                        # The choice below is trying to kill the hypothesis.
                        tokens_for_model2 = [1]  # [model2.eos]
                    else:
                        # Weiran: use scores from model2 and rnnlm2.
                        output_word_text = inverse_word_dict[word_output1]
                        logging.debug("model2 is consuming word output: %s (%d)" % (output_word_text, word_output1))
                        if ("." in output_word_text) and (output_word_text not in lexicon2):
                            logging.debug("handling abbreviation %s" % output_word_text)
                            fields = output_word_text.split("_")
                            fields_remove_dot = [x.replace(".", "") for x in fields]
                            tokens_for_model2 = []
                            for x in fields_remove_dot:
                                tokens_for_model2.extend([subword_dict2[z] if z in subword_dict2 else 1 for z in lexicon2[x]])
                        else:
                            tokens_for_model2 = [subword_dict2[x] if x in subword_dict2 else 1 for x in lexicon2[output_word_text]]
                    #logging.debug("model2 is expecting tokens:")
                    #logging.debug(" ".join(["%s (%d)" % (inverse_subword_dict2[x], x) for x in tokens_for_model2]))

                    for j in range(len(tokens_for_model2)):

                        logging.debug("Using rnnlm2 ...")
                        # vy2.unsqueeze(1)
                        vy2[0] = yseq2[-1]
                        # Weiran: note that rnnlm2_state is updated.
                        rnnlm2_state, local_lm_scores2, word_output2 = rnnlm2.predict(rnnlm2_state, vy2)[0]

                        # Accept the new token.
                        new_token = tokens_for_model2[j]
                        ys_mask2 = subsequent_mask(len(yseq2)).unsqueeze(0)

                        ys2 = torch.tensor(yseq2).unsqueeze(0)
                        local_att_scores2 = model2.decoder.forward_one_step(ys2, ys_mask2, enc_output2)[0]

                        ctc2_score, ctc2_state = ctc2_prefix_score(yseq2, [new_token], ctc2_state_prev)
                        score2 += (1.0 - ctc_weight) * float(local_att_scores2[:, new_token]) \
                                       + ctc_weight * (ctc2_score[0] - ctc2_score_prev) \
                                       + recog_args.lm_weight * float(local_lm_scores2[:, new_token])
                        ctc2_score_prev = ctc2_score[0]
                        ctc2_state_prev = ctc2_state[0]

                        # Weiran: update token list and word list.
                        yseq2.append(new_token)
                        wseq2.append(word_output2)

                if True:  # word_output1 >= 0:
                    logging.debug("\n================================================\nHypothesis in model1:")
                    logging.debug(hyp['yseq'])
                    logging.debug(" ".join([inverse_subword_dict1[int(tid)] for tid in hyp['yseq']]))
                    logging.debug("model1 current word hypothesis:")
                    logging.debug(hyp['wseq'])
                    logging.debug(" ".join([inverse_word_dict[int(wid)] for wid in hyp['wseq'] if wid > 0]))
                    if word_output1 >= 0:
                        logging.debug("outputing word: %s (%d)" % (inverse_word_dict[int(word_output1)], int(word_output1)))
                    else:
                        logging.debug("Not outputing word.")
                    logging.debug("current score1=%f" % hyp['score1'])
                    logging.debug("acc. clm score=%f" % rnnlm1_state[-1])
                    logging.debug("************************************************\nHypothesis in model2:")
                    logging.debug(yseq2)
                    logging.debug(" ".join([inverse_subword_dict2[int(tid)] for tid in yseq2]))
                    logging.debug("model2 current word hypothesis:")
                    logging.debug(wseq2)
                    logging.debug(" ".join([inverse_word_dict[int(wid)] for wid in wseq2 if wid > 0]))
                    logging.debug("================================================\n")

                # Going over the beams.
                for j in six.moves.range(beam):
                    new_hyp = {}
                    new_hyp['score1'] = hyp['score1'] + float(local_best_scores[0, j])
                    new_hyp['yseq'] = list(hyp['yseq']) + [int(local_best_ids[0, j])]
                    new_hyp['wseq'] = list(hyp['wseq']) + [word_output1]
                    new_hyp['rnnlm1_prev'] = rnnlm1_state
                    new_hyp['ctc1_state_prev'] = ctc_states[joint_best_ids[0, j]]
                    new_hyp['ctc1_score_prev'] = ctc_scores[joint_best_ids[0, j]]

                    new_hyp['score2'] = score2
                    new_hyp['yseq2'] = yseq2
                    new_hyp['wseq2'] = wseq2
                    new_hyp['rnnlm2_prev'] = rnnlm2_state
                    new_hyp['ctc2_state_prev'] = ctc2_state_prev
                    new_hyp['ctc2_score_prev'] = ctc2_score_prev

                    if word_output1 >= 0:
                        new_hyp['score'] = hyp['score1'] * (1 - recog_args.model2_weight) + new_hyp['score2'] * recog_args.model2_weight + float(local_best_scores[0, j])
                    else:
                        new_hyp['score'] = hyp['score'] + float(local_best_scores[0, j])

                    if word_output1 >= 0:
                        new_hyp['score'] += word_bonus
                        new_hyp['score1'] += word_bonus
                        new_hyp['score2'] += word_bonus

                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

            hyps_best_kept = sorted(hyps_best_kept, key=lambda x: x['score'], reverse=True)[:beam]

        # Finish expanding all hypothesis at position i.
        # sort and get nbest
        hyps = hyps_best_kept
        logging.debug('number of pruned hypothes: ' + str(len(hyps)))
        if inverse_subword_dict1 is not None:
            logging.debug(
                'best hypo: ' + ''.join([inverse_subword_dict1[int(x)] for x in hyps[0]['yseq'][1:]]))

        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info('adding <eos> in the last position in the loop')
            for hyp in hyps:
                if not hyp['yseq'][-1] == model1.eos:
                    hyp['yseq'].append(model1.eos)

        # add ended hypothesis to a final list, and removed them from current hypothes
        # (this will be a problem, number of hyps < beam)
        remained_hyps = []
        for hyp in hyps:
            if hyp['yseq'][-1] == model1.eos:
                # only store the sequence that has more than minlen outputs
                if len(hyp['yseq']) > minlen:

                    if rnnlm1:  # Word LM needs to add final <eos> score
                        RNNLM_FINAL_LIST = rnnlm1.final(hyp['rnnlm1_prev'])
                        for rnnlm1_final_score, word_output1 in RNNLM_FINAL_LIST:

                            logging.debug("\nFor rnnlm1 word_output1=%d" % word_output1)

                            yseq2 = list(hyp['yseq2'])
                            wseq2 = list(hyp['wseq2'])
                            score2 = hyp['score2']
                            rnnlm2_state = hyp['rnnlm2_prev']
                            ctc2_state_prev = hyp['ctc2_state_prev']
                            ctc2_score_prev = hyp['ctc2_score_prev']

                            # We have now restricted search space to a size of ctc_beam.
                            if word_output1 >= 0:
                                if inverse_word_dict[word_output1] == "<unk>":
                                    tokens_for_model2 = [1]  # [model2.eos]
                                else:
                                    # Weiran: use scores from model2 and rnnlm2.
                                    output_word_text = inverse_word_dict[word_output1]
                                    logging.debug("model2 is consuming word output: %s (%d)" % (output_word_text, word_output1))
                                    if ("." in output_word_text) and (output_word_text not in lexicon2):
                                        logging.debug("handling abbreviation %s" % output_word_text)
                                        fields = output_word_text.split("_")
                                        fields_remove_dot = [x.replace(".", "") for x in fields]
                                        tokens_for_model2 = []
                                        for x in fields_remove_dot:
                                            tokens_for_model2.extend([subword_dict2[z] if z in subword_dict2 else 1 for z in lexicon2[x]])
                                    else:
                                        tokens_for_model2 = [subword_dict2[x] if x in subword_dict2 else 1 for x in lexicon2[output_word_text]]

                                # Weiran: force model2 to accept eos.
                                tokens_for_model2.append(model2.eos)
                                # logging.debug("model2 is expecting tokens:")
                                # logging.debug(" ".join(["%s (%d)" % (inverse_subword_dict2[x], x) for x in tokens_for_model2]))

                                for j in range(len(tokens_for_model2)):

                                    logging.debug("Using rnnlm2 ...")
                                    # vy2.unsqueeze(1)
                                    vy2[0] = yseq2[-1]
                                    # Weiran: note that rnnlm2_state is updated.
                                    rnnlm2_state, local_lm_scores2, word_output2 = rnnlm2.predict(rnnlm2_state, vy2)[0]

                                    new_token = tokens_for_model2[j]
                                    ys_mask2 = subsequent_mask(len(yseq2)).unsqueeze(0)
                                    ys2 = torch.tensor(yseq2).unsqueeze(0)
                                    local_att_scores2 = model2.decoder.forward_one_step(ys2, ys_mask2, enc_output2)[0]

                                    ctc2_score, ctc2_state = ctc2_prefix_score(yseq2, [new_token], ctc2_state_prev)
                                    score2 += (1.0 - ctc_weight) * float(local_att_scores2[:, new_token]) \
                                              + ctc_weight * (ctc2_score[0] - ctc2_score_prev) \
                                              + recog_args.lm_weight * float(local_lm_scores2[:, new_token])
                                    ctc2_score_prev = ctc2_score[0]
                                    ctc2_state_prev = ctc2_state[0]

                                    yseq2.append(new_token)
                                    wseq2.append(word_output2)

                                rnnlm2_final_score, word_output2 = rnnlm2.final(rnnlm2_state)[0]
                                score2 += recog_args.lm_weight * float(rnnlm2_final_score)
                                wseq2.append(word_output2)

                            if True:
                                logging.debug("\n================================================\nHypothesis in model1 ending:")
                                logging.debug(hyp['yseq'])
                                logging.debug(" ".join([inverse_subword_dict1[int(tid)] for tid in hyp['yseq']]))
                                logging.debug("current word hypothesis:")
                                logging.debug(hyp['wseq'])
                                logging.debug(" ".join([inverse_word_dict[int(wid)] for wid in hyp['wseq'] if wid > 0]))
                                if word_output1 >= 0:
                                    logging.debug("outputing word: %s" % inverse_word_dict[int(word_output1)])
                                logging.debug("current score1=%f" % hyp['score1'])
                                logging.debug("adding last word+end score=%f" % (recog_args.lm_weight * rnnlm1_final_score))
                                logging.debug("************************************************\nHypothesis in model2:")
                                logging.debug(yseq2)
                                logging.debug(" ".join([inverse_subword_dict2[int(tid)] for tid in yseq2]))
                                logging.debug("model2 current word hypothesis:")
                                logging.debug(wseq2)
                                logging.debug(" ".join([inverse_word_dict[int(wid)] for wid in wseq2 if wid > 0]))
                                logging.debug("================================================")

                            new_hyp = {}
                            new_hyp['score1'] = hyp['score1'] + recog_args.lm_weight * rnnlm1_final_score
                            new_hyp['yseq'] = hyp['yseq']
                            new_hyp['wseq'] = list(hyp['wseq']) + [word_output1]

                            new_hyp['score2'] = score2
                            new_hyp['yseq2'] = yseq2
                            new_hyp['wseq2'] = wseq2

                            # Final score combines complete scores from both models.
                            new_hyp['score'] = new_hyp['score1'] * (1 - recog_args.model2_weight) + new_hyp['score2'] * recog_args.model2_weight

                            if word_output1 >= 0:
                                new_hyp['score'] += word_bonus
                                new_hyp['score1'] += word_bonus
                                new_hyp['score2'] += word_bonus
                            ended_hyps.append(new_hyp)
            else:
                remained_hyps.append(hyp)

        # end detection
        from espnet.nets.e2e_asr_common import end_detect
        if end_detect(ended_hyps, i) and recog_args.maxlenratio == 0.0:
            logging.info('end detected at %d' % i)
            break

        hyps = remained_hyps
        if len(hyps) > 0:
            logging.debug('remained hypothes: ' + str(len(hyps)))
        else:
            logging.info('no hypothesis. Finish decoding.')
            break

        if inverse_subword_dict1 is not None:
            for hyp in hyps:
                logging.debug(
                    'hypo: ' + ''.join([inverse_subword_dict1[int(x)] for x in hyp['yseq'][1:]]))

        logging.debug('number of ended hypothes: ' + str(len(ended_hyps)))
    # Finishing position i.

    # Finished label-synchronous decoding.
    nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:min(len(ended_hyps), recog_args.nbest)]

    # check number of hypothesis
    if len(nbest_hyps) == 0:
        logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
        # should copy because Namespace will be overwritten globally
        from argparse import Namespace
        recog_args = Namespace(**vars(recog_args))
        recog_args.minlenratio = max(0.0, recog_args.minlenratio - 0.1)
        return recognize_with_two_transformers(x, recog_args, model1, rnnlm1, inverse_subword_dict1, model2, rnnlm2, lexicon2, inverse_subword_dict2, inverse_word_dict)

    logging.info('total log probability for model1: ' + str(nbest_hyps[0]['score1']))
    logging.info('normalized log probability for model1: ' + str(nbest_hyps[0]['score1'] / len(nbest_hyps[0]['yseq'])))
    logging.info('total log probability for model2: ' + str(nbest_hyps[0]['score2']))
    logging.info('normalized log probability for model2: ' + str(nbest_hyps[0]['score2'] / len(nbest_hyps[0]['yseq2'])))
    return nbest_hyps
