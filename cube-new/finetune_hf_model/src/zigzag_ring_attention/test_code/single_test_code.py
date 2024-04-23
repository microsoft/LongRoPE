import torch
from examples.zigzag_ring_attention.zigzag_attn import wrap_zigzag_attn_func
 
def myrepeat(_in, repeats=None):
    return _in.repeat(*repeats)
 
class TestModule(torch.nn.Module):
    def __init__(self):
        super(TestModule, self).__init__()
 
    def forward(self, _in0, _in1, _in2):
        # Add clone to resolve the issue:
        # a leaf Variable that requires grad is being used in an in-place operation.
        _in0, _in1, _in2 = _in0.clone(), _in1.clone(), _in2.clone()
        _out0 = wrap_zigzag_attn_func(_in0, _in1, _in2, causal=True)
        # torch.save(_out0, 'zigzag_single_output.pt')
        out = 0
        for one_out in [_out0]:
            if not isinstance(one_out, torch.Tensor):
                continue
            out += torch.sum(one_out)
        return out
 
model = TestModule() #.to(torch.float16)
 
_in0 = torch.randn(1, 16 * 1024, 32, 128, dtype=torch.float16, device='cuda', requires_grad=True)
_in1 = torch.randn(1, 16 * 1024, 32, 128, dtype=torch.float16, device='cuda', requires_grad=True)
_in2 = torch.randn(1, 16 * 1024, 32, 128, dtype=torch.float16, device='cuda', requires_grad=True)
torch.save([_in0, _in1, _in2], 'zigzag_single_input.pt')
# _in0, _in1, _in2 = torch.load('zigzag_single_input.pt')
_in0 = _in0.to('cuda')
_in1 = _in1.to('cuda')
_in2 = _in2.to('cuda')

model = model.cuda()
single_loss = model(_in0, _in1, _in2)
single_loss.backward()
grad_tensors = [_in0.grad, _in1.grad, _in2.grad]
torch.save([grad_tensors, single_loss], 'zigzag_single.pt')
print('single gpu loss: ', single_loss)