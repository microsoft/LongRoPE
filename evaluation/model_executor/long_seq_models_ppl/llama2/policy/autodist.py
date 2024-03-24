import json
import time
import os
import torch
import logging

from cube.ir.operator import IRDataOperation, IRFwOperation
from cube.graph.function.anchor import IRGraphAnchor
from cube.graph.function.pyfunc import IRPyFunc

from autodist.task_config import build_argparser, TaskConfig
from autodist.apis import calc_parallel_plan, read_plan
from autodist.cube_util import replica, partition_node

_logger = logging.getLogger(__name__)

def autodist(graph, resource):
    parser = build_argparser()
    args = parser.parse_args([])

    mesh_row = 1
    mesh_col = resource.ngpus
    use_prev = False # cfg.distributed_training.autodist_use_prev_plan
    ngpus = mesh_row * mesh_col
    args.mesh_row = 1
    args.mesh_col = mesh_col
    args.micro_batch_size = 1
    args.global_batch_size = 1
    args.model = f'L'
    args.task_name = f'{args.model}_{mesh_col}gpus_{args.micro_batch_size}mbs'
    args.is_train = False
    # set to 0.5 MB
    args.ignore_small_tensor_threshold = 524288
    args.memory_granularity = 524288
    args.consider_mem = True
    # if cfg.distributed_training.autodist_mem_constraint == -1:
    if True:
        # consider memory fragmentation and other buffers, use 80% of the memory
        args.memory_constraint = int(0.8 * torch.cuda.mem_get_info()[1] / 1024 / 1024 / 1024)
    else:
        args.memory_constraint = cfg.distributed_training.autodist_mem_constraint
    args.fp16 = True
    args.verbose = 0
    args.re_profile = False
    args.world_size = resource.ngpus
    if bool(int(os.environ.get('USE_ZERO', default=0))):
        args.zero_stage = 1
        args.zero_ngroups = int(os.environ.get('ZERO_NUM_GROUPS', default=1))
    # if cfg.distributed_training.autodist_plan_save_dir:
    args.save_folder = "/root/yuanyuanxu/plan/"
    task_config = TaskConfig(**vars(args))
    if not os.path.isdir(task_config.save_folder):
        os.makedirs(task_config.save_folder)

    if not use_prev:
        compile_start_time = time.time()
        topk_plans = calc_parallel_plan(graph, task_config)
        compile_cost_time = time.time() - compile_start_time

        plan_info = []
        for plan in topk_plans:
            cur_info = {}
            assert not task_config.pipeline
            cur_spmd_desc, cur_mem, cur_time, cur_inner_time = plan
            cur_info['plan'] = cur_spmd_desc.to_json_object()
            cur_info['estimated time'] = cur_time
            cur_info['estimated inner time'] = cur_inner_time
            cur_info['estimated memory'] = cur_mem
            cur_info['compile time'] = compile_cost_time
            plan_info.append(cur_info)

        with open(task_config.backup_fname, 'w') as f:
            json.dump(plan_info, f, indent=2)
    else:
        assert os.path.exists(task_config.backup_fname)

    spmd_descs = read_plan(task_config, 0)
    assert len(spmd_descs) == 1, 'support spmd only'

    # manually control recompute for some nodes, helpful when sequence is long and flash-attn is not available
    # selected_nodes = []
    # cid2node = {}
    # for node in graph.nodes():
    #     if isinstance(node, IRFwOperation):
    #         cid2node[node.cid] = node
    #         if 'linear' in node.signature or 'bmm' in node.signature or 'matmul' in node.signature:
    #             selected_nodes.append(node)
    # layer_step = 9
    # for i in range(0, len(selected_nodes), layer_step):
    #     if i + layer_step > len(selected_nodes): break
    #     start_id = selected_nodes[i + 3].cid
    #     end_id = selected_nodes[i + 5].cid
    #     recompute_nodes = []
    #     for j in range(start_id + 1, end_id + 1):
    #         if j in cid2node:
    #             recompute_nodes.append(cid2node[j])
    #     graph.recompute(recompute_nodes)

    # check multiref before running
    for ftensor in graph.full_tensors():
        if ftensor.is_grad(): continue
        if len(graph.consumers(ftensor)) <= 1: continue
        consumers, ctensors = graph.consumers(ftensor), graph.ctensors(ftensor)
        splits = set()
        exist_cnt = 0
        for consumer, ctensor in zip(consumers, ctensors):
            if consumer.cid in spmd_descs[0].partition_descs:
                exist_cnt = exist_cnt + 1
                split = str(spmd_descs[0].partition_descs[consumer.cid].desc)
                splits.add(split)
        if len(splits) > 1 or exist_cnt != len(consumers):
            _logger.info(f'add multiref {consumers}')
            graph.multiref(ftensor)

    verbose_print = True
    def decide_print(node):
        selected_names = ['linear', 'bmm', 'matmul', 'norm']
        for name in selected_names:
            if name in node.signature:
                return True
        return False
    devs = list(range(ngpus))
    for node in graph.nodes():
        found = False
        if isinstance(node, (IRFwOperation, IRDataOperation)):
            for stage_id, spmd_desc in enumerate(spmd_descs):
                if node.cid in spmd_desc.partition_descs:
                    if verbose_print or decide_print(node):
                        _logger.info(f'partition {node} with anno {node.anno}, plan: {spmd_desc.partition_descs[node.cid].desc}')
                    partition_node(node, graph, devs,
                                   spmd_desc.partition_descs[node.cid])
                    found = True
            if not found:
                if not isinstance(node, (IRGraphAnchor, IRPyFunc)):
                    replica(graph, node, devs)

    return graph