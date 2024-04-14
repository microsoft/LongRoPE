import torch
import cube
from cube.graph import IRGraph
from cube.ir.operator import IRFwOperation
from cube.parallel import parallelize, ComputeConfig, ReuseType
import torch.distributed
 
import cube.graph
import cube.graph.function
from examples.zigzag_ring_attention.zigzag_attn import wrap_zigzag_attn_func
# from cube.graph.function.zigzag_attn import wrap_zigzag_attn_func

class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
 
    def forward(self, _in0, _in1, _in2):
        # Add clone to resolve the issue:
        # a leaf Variable that requires grad is being used in an in-place operation.
        _in0, _in1, _in2 = _in0.clone(), _in1.clone(), _in2.clone()
        _out0 = wrap_zigzag_attn_func(_in0, _in1, _in2)
        out = 0
        for one_out in [_out0]:
            if not isinstance(one_out, torch.Tensor):
                continue
            out += torch.sum(one_out)
        return out
 
model = TestModule() #.to(torch.float16)
 
cube.init()
rank_id = torch.distributed.get_rank()
    
def policy(graph: IRGraph, resource: ComputeConfig) -> IRGraph:
    ngpus = resource.plan_ngpus
    partitioned = False
    for idx, node in enumerate(graph.select(ntype=IRFwOperation)):
        # if not partitioned and node.signature == 'cube.graph.function.zigzag_attn.wrap_zigzag_attn_func':
        if not partitioned and node.signature == 'examples.zigzag_ring_attention.zigzag_attn.wrap_zigzag_attn_func':
            print('Partitioned node: ', node)
            sub_nodes = graph.partition(
                node, node.algorithms('dim'), idx=0, dim=1, num=ngpus)
            partitioned = True
        else:
            sub_nodes = graph.replicate(node, times=ngpus)
        for idx, sub_node in enumerate(sub_nodes):
            graph.assign(sub_node, idx)
    assert partitioned, f'No node is partitioned for torch.conv2d.'
    return graph

# _in0 = torch.randn(1, 32, 32, 128, dtype=torch.float16, device='cuda', requires_grad=True)
# _in1 = torch.randn(1, 32, 32, 128, dtype=torch.float16, device='cuda', requires_grad=True)
# _in2 = torch.randn(1, 32, 32, 128, dtype=torch.float16, device='cuda', requires_grad=True)
_in0, _in1, _in2 = torch.load('zigzag_single_input.pt')
_in0 = _in0.to('cuda')
_in1 = _in1.to('cuda')
_in2 = _in2.to('cuda')
_in0.retain_grad()
_in1.retain_grad()
_in2.retain_grad()

parallel_model = parallelize(model, dummy_input={"_in0": _in0, "_in1": _in1, "_in2": _in2}, pas_policy=policy,
                             compute_config=ComputeConfig(8, 8), reuse=ReuseType.OVERRIDE)
parallel_model = parallel_model.cuda()

print("begin training")

parallel_model.train()

_in0 = torch.randn(1, 16 * 1024, 32, 128, dtype=torch.float16, device='cuda', requires_grad=True)
_in1 = torch.randn(1, 16 * 1024, 32, 128, dtype=torch.float16, device='cuda', requires_grad=True)
_in2 = torch.randn(1, 16 * 1024, 32, 128, dtype=torch.float16, device='cuda', requires_grad=True)
# _in0, _in1, _in2 = torch.load('zigzag_single_input.pt')
_in0 = _in0.to('cuda')
_in1 = _in1.to('cuda')
_in2 = _in2.to('cuda')
_in0.retain_grad()
_in1.retain_grad()
_in2.retain_grad()
 
para_loss = parallel_model(_in0, _in1, _in2)
para_loss.backward()
parallel_model.sync_grad()
grad_tensors = [_in0.grad, _in1.grad, _in2.grad]
torch.save([grad_tensors, para_loss], 'zigzag_'+str(rank_id)+'.pt')
print('multiple gpus loss: ', para_loss)

