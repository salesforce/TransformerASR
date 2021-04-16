#!/usr/bin/env python3
# encoding: utf-8

"""
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

# Copyright 2020 Weiran Wang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Training/decoding definition for the speech recognition task."""
import pdb
import json
import logging
import math
import os

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

from espnet.asr.pytorch_backend.asr_dsl import _recursive_to, CustomConverter
from espnet.asr.pytorch_backend.asr_dsl import torch_resume, torch_snapshot
from espnet.asr.pytorch_backend.asr_dsl import multiply_grads


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

    if (args.enc_init is not None or args.dec_init is not None or args.ctc_init is not None):
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
        train_dict = dict({"epoch": 0, "current_position": 0, "iteration": 0, "validation_loss": []})

    model = DistributedModel(model, args.local_rank, bucket_cap_mb=256, find_unused_parameters=True)

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
    loss_ctc_train = 0.0
    loss_realigner_train = 0.0
    loss_masked_train = 0.0

    while True:

        if 0 < args.epochs <= epoch:
            train_loader.close()
            valid_loader.close()
            break

        # Adjust temperature.
        model.tau = args.error_temperature * args.temperature_decay_rate ** max(0, epoch - args.temperature_decay_epoch)

        with model.no_sync():
            # Only synchronize the gradients every accum_grad steps, or at the end of an epoch
            while forward_count < args.accum_grad - 1 and train_loader.current_position < len(train_loader)-1:

                # Get the next batch
                logging.info("Getting batch from dataloader")
                batch = train_loader.next()
                logging.info("CHECK I: position=%d, total=%d, epoch=%d, model_device=%s" % (
                train_loader.current_position, len(train_loader), train_loader.epoch, next(model.parameters()).device))
                x = _recursive_to(train_converter(batch), device)
                logging.info("Move batch to GPU")

                # Weiran: the actual number of utts in the minibatch.
                x_num = x[0].size(0)
                y_num = float(torch.sum(x[2] != train_converter.ignore_id).cpu())

                loss, loss_data, loss_ctc_data, loss_realigner_data, loss_masked_data, _, err_greedy = model(*x)

                if global_rank==0:
                    logging.info("x input lengths:")
                    logging.info(x[1].cpu())
                    logging.info("label lengths:")
                    logging.info(torch.sum(x[2] != train_converter.ignore_id, dim=1).cpu())
                    logging.info("realigner errors, greedy:")
                    logging.info(err_greedy)

                xnum_train += x_num
                ynum_train += y_num
                loss_train += x_num * loss_data
                loss_ctc_train += x_num * loss_ctc_data
                loss_realigner_train += x_num * loss_realigner_data
                loss_masked_train += x_num * loss_masked_data

                # loss is the sum, not average.
                loss = x_num * loss
                loss.backward()
                loss.detach()  # Truncate the graph

                forward_count += 1

        # Perform the same loop as above, but sync the gradients if it's been
        # accum_grad steps, or the epoch is about to finish
        logging.info("Getting batch from dataloader")
        batch = train_loader.next()
        logging.info("CHECK II: position=%d, total=%d, epoch=%d, model_device=%s" % (train_loader.current_position, len(train_loader), train_loader.epoch, next(model.parameters()).device))
        x = _recursive_to(train_converter(batch), device)
        logging.info("Move batch to GPU")

        # Weiran: the actual number of utts in the minibatch.
        x_num = x[0].size(0)
        y_num = float(torch.sum(x[2] != train_converter.ignore_id).cpu())
        loss, loss_data, loss_ctc_data, loss_realigner_data, loss_masked_data, _, err_greedy = model(*x)

        if global_rank == 0:
            logging.info("x input lengths:")
            logging.info(x[1].cpu())
            logging.info("label lengths:")
            logging.info(torch.sum(x[2] != train_converter.ignore_id, dim=1).cpu())
            logging.info("realigner errors, greedy:")
            logging.info(err_greedy)

        xnum_train += x_num
        ynum_train += y_num
        loss_train += x_num * loss_data
        loss_ctc_train += x_num * loss_ctc_data
        loss_realigner_train += x_num * loss_realigner_data
        loss_masked_train += x_num * loss_masked_data

        # loss is the sum, not average.
        loss = x_num * loss
        loss.backward()
        loss.detach()  # Truncate the graph

        # update parameters
        forward_count += 1

        is_new_epoch = (not epoch == train_loader.epoch)
        assert is_new_epoch or forward_count == args.accum_grad

        # Needed when distributed_world_size > 1.
        xnum_all, ynum_all, loss_all, loss_masked_all, loss_ctc_all, loss_realigner_all = \
            zip(*all_gather_list([xnum_train, ynum_train, loss_train, loss_masked_train, loss_ctc_train, loss_realigner_train]))
        total_xnum = sum(xnum_all)
        total_ynum = sum(ynum_all)
        total_loss = sum(loss_all)
        total_loss_masked = sum(loss_masked_all)
        total_loss_ctc = sum(loss_ctc_all)
        total_loss_realigner = sum(loss_realigner_all)

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

        forward_count = 0
        iteration += 1
        if iteration % args.report_interval_iters == 0 and global_rank == 0:
            reporter.add_scalar("train/loss", total_loss / total_xnum, iteration)
            reporter.add_scalar("train/loss_ctc", total_loss_ctc / total_xnum, iteration)
            reporter.add_scalar("train/loss_realigner", total_loss_realigner / total_xnum, iteration)
            reporter.add_scalar("train/loss_masked", total_loss_masked / total_xnum, iteration)
            reporter.add_scalar("train/lr", optimizer.param_groups[0]['lr'], iteration)

        # Reset the accumulation again.
        xnum_train = 0
        ynum_train = 0
        loss_train = 0.0
        loss_masked_train = 0.0
        loss_ctc_train = 0.0
        loss_realigner_train = 0.0

        # Finished one epoch.
        if is_new_epoch:
            epoch += 1

            # Evaluate the model with the validation dataset at the end of every epoch.
            if global_rank == 0:
                logging.info("Start validation for epoch %d" % train_loader.epoch)

            model.eval()

            xnum_valid = 0
            ynum_valid = 0
            loss_valid = 0.0
            loss_masked_valid = 0.0
            loss_ctc_valid = 0.0
            err_ctc_valid = 0.0
            err_realigner_valid = 0.0
            loss_realigner_valid = 0.0

            for batch in valid_loader:

                logging.info("Getting batch from dataloader")
                x = _recursive_to(valid_converter(batch), device)
                logging.info("Move batch to GPU")

                x_num = x[0].size(0)
                y_num = float(torch.sum(x[2] != valid_converter.ignore_id).cpu())

                with torch.no_grad():
                    _, loss_data, loss_ctc_data, loss_realigner_data, loss_masked_data, err_ctc_data, err_greedy = model(*x)

                xnum_valid += x_num
                ynum_valid += y_num
                loss_valid += x_num * loss_data
                loss_masked_valid += x_num * loss_masked_data
                loss_ctc_valid += x_num * loss_ctc_data
                loss_realigner_valid += x_num * loss_realigner_data
                err_ctc_valid += torch.sum(err_ctc_data)
                err_realigner_valid += torch.sum(err_greedy)

            xnum_all, ynum_all, loss_all, loss_masked_all, loss_ctc_all, err_ctc_all, err_realigner_all, loss_realigner_all = \
                zip(*all_gather_list([xnum_valid, ynum_valid, loss_valid, loss_masked_valid, loss_ctc_valid, err_ctc_valid, err_realigner_valid, loss_realigner_valid]))

            total_xnum = sum(xnum_all)
            total_ynum = sum(ynum_all)
            total_loss = sum(loss_all)
            total_loss_masked = sum(loss_masked_all)
            total_loss_ctc = sum(loss_ctc_all)
            total_err_ctc = sum(err_ctc_all)
            total_err_realigner = sum(err_realigner_all)
            total_loss_realigner = sum(loss_realigner_all)

            # Each GPU has access to the validation loss in order to adjust learning rate.
            validation_loss.append(total_loss / total_xnum)
            if global_rank == 0:
                reporter.add_scalar("valid/loss", total_loss / total_xnum, iteration)
                reporter.add_scalar("valid/loss_ctc", total_loss_ctc / total_xnum, iteration)
                reporter.add_scalar("valid/loss_realigner", total_loss_realigner / total_xnum, iteration)
                reporter.add_scalar("valid/loss_masked", total_loss_masked / total_xnum, iteration)
                reporter.add_scalar("valid/err_ctc", total_err_ctc / total_ynum, iteration)
                reporter.add_scalar("valid/err_realigner", total_err_realigner / total_ynum, iteration)

            # Save model in each epoch.
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
