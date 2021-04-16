############################## DISTRIBUTED UTILS ################################

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import pickle
import torch
# from collections import namedtuple


def is_master(distributed_rank):
    return distributed_rank==0


# _use_c10d = [True]
#
# C10dStatus = namedtuple('C10dStatus', ['has_c10d', 'is_default'])
#
# if hasattr(nn.parallel, 'deprecated'):
#     c10d_status = C10dStatus(has_c10d=True, is_default=True)
# elif hasattr(torch.distributed, 'c10d') and hasattr(torch.distributed.c10d,
#         'init_process_group'):
#     c10d_status = C10dStatus(has_c10d=True, is_default=False)
# else:
#     c10d_status = C10dStatus(has_c10d=False, is_default=False)
#
# if c10d_status.is_default:
#     import torch.distributed as dist_c10d
#     import torch.distributed.deprecated as dist_no_c10d
# elif c10d_status.has_c10d:
#     import torch.distributed.c10d as dist_c10d
#     import torch.distributed as dist_no_c10d
# else:
#     import torch.distributed as dist_no_c10d


def distributed_init(distributed_world_size,
                     distributed_rank,
                     distributed_init_method,
                     distributed_backend,
                     multi_node
                     ):
    if distributed_world_size==1:
        raise ValueError(
            'Cannot initialize distributed with distributed_world_size=1')

    print('| distributed init (rank {}): {}'.format(
        distributed_rank, distributed_init_method), flush=True)

    init_fn = torch.distributed.init_process_group

    if multi_node:
        init_fn(
            backend=distributed_backend,
            init_method="env://"
        )
    else:
        init_fn(
            backend=distributed_backend,
            init_method=distributed_init_method,
            world_size=distributed_world_size,
            rank=distributed_rank,
        )

    suppress_output(is_master(distributed_rank))

    return distributed_rank


def suppress_output(is_master):
    """Suppress printing on the current device. Force printing with
    `force=True`."""
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def get_rank():
    return torch.distributed.get_rank()
    # if _use_c10d[0]:
    #     return dist_c10d.get_rank()
    # else:
    #     return dist_no_c10d.get_rank()


def get_world_size():
    return torch.distributed.get_world_size()
    #if _use_c10d[0]:
    #    return dist_c10d.get_world_size()
    #else:
    #    return dist_no_c10d.get_world_size()


def get_default_group():
    return torch.distributed.group.WORLD
    # if _use_c10d[0]:
    #     return dist_c10d.group.WORLD
    # else:
    #     return dist_no_c10d.group.WORLD


def all_reduce(tensor, group=None):
    if group is None:
        group = get_default_group()
    return torch.distributed.all_reduce(tensor, group=group)
    # if _use_c10d[0]:
    #     return dist_c10d.all_reduce(tensor, group=group)
    # else:
    #     return dist_no_c10d.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    rank = get_rank()
    world_size = get_world_size()

    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 3 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size <= 255 * 255 * 255

    cpu_buffer[0] = (enc_size // 255) // 255  # this encoding works for max_size < 65k
    cpu_buffer[1] = (enc_size // 255) % 255
    cpu_buffer[2] = enc_size % 255
    #print(item(cpu_buffer[0])*255*255+item(cpu_buffer[1])*255+item(cpu_buffer[2]), enc_size)
    cpu_buffer[3: enc_size + 3] = torch.ByteTensor(list(enc))
    start = rank * max_size
    size = enc_size + 3
    buffer[start: start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = (255*255 * item(out_buffer[0])) + (255 * item(out_buffer[1])) + item(out_buffer[2])
            if size > 0:
                result.append(
                    pickle.loads(bytes(out_buffer[3: size + 3].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list '
            'requires all '
            'workers to enter the function together, so this error usually '
            'indicates '
            'that the workers have fallen out of sync somehow. Workers can '
            'fall out of '
            'sync if one of them runs out of memory, or if there are other '
            'conditions '
            'in your training script that can cause one worker to finish an '
            'epoch '
            'while other workers are still iterating over their portions of '
            'the data.'
        )


def all_gather_list_old(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.

    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.

    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
        max_size (int, optional): maximum size of the data to be gathered
            across workers
    """
    rank = get_rank()
    world_size = get_world_size()

    buffer_size = max_size * world_size
    if not hasattr(all_gather_list, '_buffer') or \
            all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()
    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    enc = pickle.dumps(data)
    enc_size = len(enc)
    if enc_size + 2 > max_size:
        raise ValueError(
            'encoded data exceeds max_size: {}'.format(enc_size + 2))
    assert max_size <= 255 * 255 * 255

    cpu_buffer[0] = (enc_size // 255) // 255  # this encoding works for max_size < 65k
    cpu_buffer[1] = (enc_size // 255) % 255
    cpu_buffer[2] = enc_size % 255
    #print(item(cpu_buffer[0])*255*255+item(cpu_buffer[1])*255+item(cpu_buffer[2]), enc_size)
    cpu_buffer[3: enc_size + 3] = torch.ByteTensor(list(enc))
    start = rank * max_size
    size = enc_size + 3
    buffer[start: start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size: (i + 1) * max_size]
            size = (255*255 * item(out_buffer[0])) + (255 * item(out_buffer[1])) + item(out_buffer[2])
            if size > 0:
                result.append(
                    pickle.loads(bytes(out_buffer[3: size + 3].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            'Unable to unpickle data from other workers. all_gather_list '
            'requires all '
            'workers to enter the function together, so this error usually '
            'indicates '
            'that the workers have fallen out of sync somehow. Workers can '
            'fall out of '
            'sync if one of them runs out of memory, or if there are other '
            'conditions '
            'in your training script that can cause one worker to finish an '
            'epoch '
            'while other workers are still iterating over their portions of '
            'the data.'
        )


def item(tensor):
    if hasattr(tensor, 'item'):
        return tensor.item()
    if hasattr(tensor, '__getitem__'):
        return tensor[0]
    return tensor



# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.


def DistributedModel(model, device_id, bucket_cap_mb, find_unused_parameters=False):
    """
    Wrap a *model* to support distributed data parallel training.
    This is similar to the built-in DistributedDataParallel, but allows
    additional configuration of the DistributedDataParallel class to
    use, and also provides easier access to the wrapped model by
    forwarding requests for missing attributes to the wrapped model.
    Args:
        args (argparse.Namespace): fairseq args
        model (BaseFairseqModel): model to wrap
    """

    # determine which DDP class to extend
    ddp_class = torch.nn.parallel.DistributedDataParallel

    init_kwargs = dict(
        module=model,
        # gradient_average=False,
        bucket_cap_mb=bucket_cap_mb,
        device_ids=[device_id],
        output_device=device_id,
        find_unused_parameters=find_unused_parameters
    )

    class _DistributedFairseqModel(ddp_class):
        """Extend DistributedDataParallel to check for missing
        attributes in the wrapped module."""

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __getattr__(self, name):
            wrapped_module = super().__getattr__('module')
            if hasattr(wrapped_module, name):
                return getattr(wrapped_module, name)
            return super().__getattr__(name)

    return _DistributedFairseqModel(**init_kwargs)
