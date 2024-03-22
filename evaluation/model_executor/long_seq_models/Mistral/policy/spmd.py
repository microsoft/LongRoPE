from typing import List, Union, Callable, Optional, Tuple
import logging

import cube
from cube.graph import IRGraph
from cube.graph.segment import IRSegment
from cube.graph.function.dimops import IRDimops
from cube.graph.gener.rvd.intra import IntraAutoPlacer
from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.ir.cten import IRCell
from cube.ir.tensor import IRFullTensor
from cube.graph.function.anchor import IRGraphAnchor
from cube.utils import print_each_rank
from cube.ir.tensor import IRSubTensor

import numpy as np

_logger = logging.getLogger(__name__)
from cube.ir.operator import IRBpOperation, IRDataOperation, IRFwOperation


def PASSingle(graph: IRGraph, resource):
    assert resource.ngpus == 1
    print(graph.extra_repr())
    for node in graph.nodes():
        if not isinstance(node, IRBpOperation):
            graph.assign(node, 0)
    return graph

''' Parallelism for llama model'''

def _tp_autoplace(segment: IRSegment, ftensor: IRFullTensor,
                  producers: List[IRFwOperation], devs: List[int],
                  sub_nodes: List[IRFwOperation]) -> List[int]:
    """decide the devices of the partitioned `sub-nodes` to achieve optimal communication
    
    Args:
        segment (IRSegment): segment of the ftensor
        ftensor (IRFullTensor): the tensor to be partitioned
        producers (List[IRFwOperation]): producers of the ftensor
        devs (List[int]): devices to be placed
        sub_nodes (List[IRFwOperation]): partitioned nodes

    Returns:
        List[int]: devices of the partitioned `sub-nodes`
    """
    if ftensor.is_param() or len(producers) != len(sub_nodes):
        _logger.warning(f"skip auto placer due to condition not matched: "
                        f"nproducers: {len(producers)}, nconsumers: {len(sub_nodes)}, "
                        f"producer name: {producers[0].name if len(producers) > 0 else None}")
        devs = sorted(list(devs))
    else:
        devs = IntraAutoPlacer.auto_place(segment, ftensor, producers, sub_nodes)
    return devs

# tensor parallelism
def tensor_parallelism(graph: IRGraph, node: IRDimops, 
                       idx: int, dim: int, devs: List[int],
                       autoplace: bool = False) -> List[IRDimops]:
    """Apply tensor parallelism of a node to devs"""
    if len(devs) == 1:
        graph.assign(node, devs[0])
        return [node]
    # transformation
    algo = node.algorithms('dim')
    sub_nodes = graph.partition(node, algo, idx=idx, dim=dim, num=len(devs))
    assert sub_nodes is not None

    if autoplace:
        segment = graph.segment(node)
        devs = _tp_autoplace(segment, node.input(idx).parent,
                             segment.producers(node.input(idx).parent),
                             devs, sub_nodes)
    # assign
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes


# replica
def replica(graph: IRGraph, node: Union[IRFwOperation, IRDataOperation], 
            devs: List[int]) -> List[Union[IRFwOperation, IRDataOperation]]:
    """Replicate a forward node or dataloader to devs"""
    if len(devs) == 1:
        graph.assign(node, devs[0])
        return [node]
    sub_nodes = graph.replicate(node, times=len(devs))
    for devid, sub_node in zip(devs, sub_nodes):
        graph.assign(sub_node, devid)
    return sub_nodes

def PASMegatronTP(graph: IRGraph, resource, **kwargs):
    """Megatron-way tensor parallelism"""
    devs = list(range(resource.ngpus))
    print(f"tensor parallelism devs: {devs}")
    
    # partition embed √
    # [bs, seq_length] [vocab_size, hidden_size] -> [bs, seq_length, hidden_size]
    # NOTE: padding?
    # for embed in graph.select(name='embedding'):
    #     tensor_parallelism(graph, embed, idx=1, dim=0, devs=devs)
        
    # feedforward √
    # bs seq_len hidden_size^, inter_size+ hidden_size^, \
    # inter_size+ hidden_size^, hidden_size^ inter_size+ -> bs seq_len^ hidden_size^
    for embed in graph.select(name='ffn'):
        tensor_parallelism(graph, embed, idx=1, dim=0, devs=devs)

    # apply_rotary_pos_emb_inference √
    for embed in graph.select(name='apply_rotary_pos_emb_inference'):
        tensor_parallelism(graph, embed, idx=0, dim=2, devs=devs)
    
    # norm √
    for attn in graph.select(name='RmsNorm'):
        tensor_parallelism(graph, attn, idx=0, dim=1, devs=devs)
        
    # partition linear 
    '''
    q: [bs, seq_length, hidden_size] [hidden_size, hidden_size^] -> [bs, seq_length, hidden_size]
    k: [bs, seq_length, hidden_size] [hidden_size, hidden_size^] -> [bs, seq_length, hidden_size]
    v: [bs, seq_length, hidden_size] [hidden_size, hidden_size^] -> [bs, seq_length, hidden_size]
    out: [bs, seq_length, hidden_size] [hidden_size^, hidden_size] -> [bs, seq_length, hidden_size]
    
    gate: [bs, seq_length, hidden_size] [hidden_size, intermediate_size] -> [bs, seq_length, intermediate_size]
    up: [bs, seq_length, hidden_size] [hidden_size, intermediate_size] -> [bs, seq_length, intermediate_size]
    down: [bs, seq_length, intermediate_size] [intermediate_size, hidden_size] -> [bs, seq_length, hidden_size]
    '''
    for idx, linr in enumerate(graph.select(name='linear')):
        if idx < 128:
            if idx % 4 == 3: # o_proj
                tensor_parallelism(graph, linr, idx=1, dim=1, devs=devs)
            else:
                tensor_parallelism(graph, linr, idx=1, dim=0, devs=devs)
        else: # loop after
            pass
            # # [bs, seq_length, hidden_size] [vocab_size, hidden_size] -> [bs, seq_length, vocab_size]
            # tensor_parallelism(graph, linr, idx=0, dim=2, devs=devs)
    
    
    # [bs, seq_length, (num_heads head_dim)] -> [bs seq_length num_heads head_dim]
    for idx, linr in enumerate(graph.select(name='view')):
        if idx is not 0 and idx < 97:
            tensor_parallelism(graph, linr, idx=0, dim=2, devs=devs)
                
    # attention √
    # (bs seq_length num_heads head_dim) (bs seq_length num_heads head_dim) 
    # (bs seq_length num_heads head_dim) -> (bs seq_length (num_heads head_dim))
    for attn in graph.select(name='flash_attention'):
        tensor_parallelism(graph, attn, idx=0, dim=2, devs=devs)
        
    # replica other nodes
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs)

    # print(graph.extra_repr())

    return graph

def PASMegatronTPCache(graph: IRGraph, resource, **kwargs):
    """Megatron-way tensor parallelism"""
    devs = list(range(resource.ngpus))
    print(f"tensor parallelism devs: {devs}")
        
    # feedforward √
    # bs seq_len hidden_size^, inter_size+ hidden_size^, \
    # inter_size+ hidden_size^, hidden_size^ inter_size+ -> bs seq_len^ hidden_size^
    for embed in graph.select(name='ffn'):
        tensor_parallelism(graph, embed, idx=1, dim=0, devs=devs)

    # apply_rotary_pos_emb_inference √
    for embed in graph.select(name='apply_rotary_pos_emb_inference'):
        tensor_parallelism(graph, embed, idx=0, dim=2, devs=devs)
    
    # norm √
    # for attn in graph.select(name='RmsNorm'):
    #     tensor_parallelism(graph, attn, idx=0, dim=1, devs=devs)
        
    # partition linear 
    '''
    q: [bs, seq_length, hidden_size] [hidden_size, hidden_size^] -> [bs, seq_length, hidden_size]
    k: [bs, seq_length, hidden_size] [hidden_size, hidden_size^] -> [bs, seq_length, hidden_size]
    v: [bs, seq_length, hidden_size] [hidden_size, hidden_size^] -> [bs, seq_length, hidden_size]
    out: [bs, seq_length, hidden_size] [hidden_size^, hidden_size] -> [bs, seq_length, hidden_size]
    '''
    for idx, linr in enumerate(graph.select(name='linear')):
        if idx < 128:
            if idx % 4 == 3: # o_proj
                tensor_parallelism(graph, linr, idx=1, dim=1, devs=devs)
            else:
                tensor_parallelism(graph, linr, idx=1, dim=0, devs=devs)
        else: # loop after
            pass
            # # [bs, seq_length, hidden_size] [vocab_size, hidden_size] -> [bs, seq_length, vocab_size]
            # tensor_parallelism(graph, linr, idx=0, dim=2, devs=devs)
    
    
    # [bs, seq_length, (num_heads head_dim)] -> [bs seq_length num_heads head_dim]
    for idx, linr in enumerate(graph.select(name='view')):
        if idx != 0 and idx < 97:
            tensor_parallelism(graph, linr, idx=0, dim=2, devs=devs)
                
    # attention √
    # (bs seq_length num_heads head_dim) (bs seq_length num_heads head_dim) 
    # (bs seq_length num_heads head_dim) -> (bs seq_length (num_heads head_dim))
    for attn in graph.select(name='flash_attention_with_kvcache'):
        tensor_parallelism(graph, attn, idx=0, dim=2, devs=devs)
        
    # replica other nodes
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs)

    # print(graph.extra_repr())

    return graph

def PAS_Sequence_Parallel(graph: IRGraph, resource, **kwargs):
    """Megatron-way tensor parallelism"""
    devs = list(range(resource.ngpus))
    print(f"tensor parallelism devs: {devs}")
            
    # feedforward √
    # bs seq_len hidden_size^, inter_size+ hidden_size^, \
    # inter_size+ hidden_size^, hidden_size^ inter_size+ -> bs seq_len^ hidden_size^
    for embed in graph.select(name='ffn'):
        tensor_parallelism(graph, embed, idx=0, dim=1, devs=devs)

    # apply_rotary_pos_emb_inference √
    for embed in graph.select(name='apply_rotary_pos_emb_inference'):
        tensor_parallelism(graph, embed, idx=0, dim=2, devs=devs)
    
    # norm √
    for attn in graph.select(name='RmsNorm'):
        tensor_parallelism(graph, attn, idx=0, dim=1, devs=devs)
        
    # add_ √ : [bs seq_len hidden_size^, bs seq_len hidden_size^] -> bs seq_len hidden_size^
    for idx, attn in enumerate(graph.select(name='add')):
        if idx > 0 and (idx - 1) % 3 > 0:
            tensor_parallelism(graph, attn, idx=0, dim=1, devs=devs)
    
    # partition linear 
    '''
    q: [bs, seq_length, hidden_size] [hidden_size, hidden_size^] -> [bs, seq_length, hidden_size]
    k: [bs, seq_length, hidden_size] [hidden_size, hidden_size^] -> [bs, seq_length, hidden_size]
    v: [bs, seq_length, hidden_size] [hidden_size, hidden_size^] -> [bs, seq_length, hidden_size]
    out: [bs, seq_length, hidden_size] [hidden_size^, hidden_size] -> [bs, seq_length, hidden_size]
    '''
    for idx, linr in enumerate(graph.select(name='linear')):
        if idx < 128:
            if idx % 4 == 3: # o_proj
                tensor_parallelism(graph, linr, idx=1, dim=1, devs=devs)
            else:
                tensor_parallelism(graph, linr, idx=1, dim=0, devs=devs)
        else: # loop after
            pass
            # # [bs, seq_length, hidden_size] [vocab_size, hidden_size] -> [bs, seq_length, vocab_size]
            # tensor_parallelism(graph, linr, idx=0, dim=2, devs=devs)
    
    # [bs, seq_length, (num_heads head_dim)] -> [bs seq_length num_heads head_dim]
    for idx, linr in enumerate(graph.select(name='view')):
        if idx is not 0 and idx < 97:
            tensor_parallelism(graph, linr, idx=0, dim=2, devs=devs)
                
    # attention √
    # (bs seq_length num_heads head_dim) (bs seq_length num_heads head_dim) 
    # (bs seq_length num_heads head_dim) -> (bs seq_length (num_heads head_dim))
    for attn in graph.select(name='flash_attention_with_kvcache'):
        tensor_parallelism(graph, attn, idx=0, dim=2, devs=devs)
        
    # replica other nodes
    for node in graph.select(ntype=(IRFwOperation, IRDataOperation)):
        if len(node.device) == 0:
            replica(graph, node, devs)

    # print(graph.extra_repr())

    return graph