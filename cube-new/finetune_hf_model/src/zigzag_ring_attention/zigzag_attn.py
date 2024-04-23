"""
Currently, this file wraps the zigzag_attention
"""
from typing import Tuple
import torch
from torch import Tensor
import torch.distributed

from cube.graph.parser.register import register
from zigzag_ring_attention.zigzag_utils.zigzag_attn_implementation import ZigZagRingFlashAttnFunc
from flash_attn import flash_attn_func

import torch.distributed as dist
from cube.runtime.device import DeviceGroup

def recover_output(to_send: torch.Tensor, 
                    process_group: dist.ProcessGroup = None):

    # NOTE: We must use outplace implementation, or with raise error at backward dur to inplace operation
    # So it seems that we can not change to_send directly, so we create a new tensor to store the result.
    to_send_f = torch.empty_like(to_send)
    
    block_seq_len = to_send.shape[1] // 2
    
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    
    # restore output sequence
    if rank >= world_size // 2:
        to_send_slice = to_send[:, :block_seq_len, ...] # 3 - 1 6
        res = to_send_f[:, block_seq_len:] # 3 - 1 6
    else:
        to_send_slice = to_send[:, block_seq_len:, ...] # 0 - 0 7
        res = to_send_f[:, block_seq_len:] # 0 - 0 7
    
    assert to_send_slice.is_contiguous()
    assert res.is_contiguous()
    
    _ops = []
    sr_rank = (world_size - rank - 1) % world_size
    send_op = dist.P2POp(
        dist.isend, to_send_slice, sr_rank, group=process_group
    )
    recv_op = dist.P2POp(
        dist.irecv, res, sr_rank, group=process_group)
    
    _ops.append(send_op)
    _ops.append(recv_op)
    
    response = dist.batch_isend_irecv(_ops)
    for resp in response:
        resp.wait()
            
    if rank >= world_size // 2: # 3 - 1 6
        to_send_f[:, :block_seq_len] = to_send[:, block_seq_len:, ...]
    else: # 0 - 0 7
        to_send_f[:, :block_seq_len] = to_send[:, :block_seq_len, ...]
        
    return to_send_f.contiguous()

def shuffle_input(to_send: torch.Tensor, 
                  process_group: dist.ProcessGroup = None):
    
    to_send_f = torch.zeros_like(to_send, dtype=to_send.dtype, device=to_send.device)

    block_seq_len = to_send.shape[1] // 2
    to_send_slice = to_send[:, block_seq_len:]

    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
        
    if rank >= world_size // 2: # 3 -- 6 7, -> 1 6
        res = to_send_f[:, :block_seq_len, ...]
    else:                       # 0 -- 0 1, -> 0 7
        res = to_send_f[:, block_seq_len:, ...]
    
    _ops = []
    sr_rank = (world_size - rank - 1) % world_size
    send_op = dist.P2POp(
        dist.isend, to_send_slice, sr_rank, group=process_group
    )
    recv_op = dist.P2POp(
        dist.irecv, res, sr_rank, group=process_group)
    
    _ops.append(send_op)
    _ops.append(recv_op)
    
    response = dist.batch_isend_irecv(_ops)
    for resp in response:
        resp.wait()

    if rank >= world_size // 2: # 3 -- 6 7, -> 1 6
        to_send_f[:, block_seq_len:] = to_send[:, :block_seq_len]
    else:                       # 0 -- 0 1, -> 0 7
        to_send_f[:, :block_seq_len] = to_send[:, :block_seq_len]
    
    return to_send_f

def wrap_zigzag_attn_func(q: Tensor, k: Tensor, v: Tensor, softmax_scale: Tensor=None,
                          dropout_p: float=0.0, causal: bool=True, window_size: Tuple[int]=(-1, -1),
                          alibi_slopes: Tensor=None, deterministic: bool=False,
                          return_attn_probs: bool=False,
                          process_group: Tuple[int]=None) -> Tensor:
    
    if process_group is None or len(process_group) == 1:
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)
        output = flash_attn_func(q, k, v, 0.0, softmax_scale, causal)
        return output
    
    assert causal == True, "zigzag_ring is meaningless for causal=False"
    assert q.shape == k.shape == v.shape, "q, k, v must have the same shape"
    assert len(q.shape) == 4, "q, k, v must have shape [bs, ql, qh, dim]"
    assert q.shape[0] == 1, "batch size must be 1 currently"
    
    # NOTE: get the local process group from tulpe process group
    local_process_group = DeviceGroup().get_group(process_group)

    #NOTE: shuffle q, k, v
    q = shuffle_input(to_send=q, process_group=local_process_group)
    k = shuffle_input(to_send=k, process_group=local_process_group)
    v = shuffle_input(to_send=v, process_group=local_process_group)
    
    output = ZigZagRingFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        local_process_group,
    ).contiguous()
    
    #NOTE: recover output sequence
    output = recover_output(output, process_group=local_process_group)

    return output
