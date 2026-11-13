import types
import torch.nn.functional as F
def linear_patch_embed(hidden_states,projection):
 if projection.groups!=1 or tuple(projection.kernel_size)!=tuple(projection.stride) or any(projection.padding) or tuple(projection.dilation)!=(1,1,1):raise ValueError('Projection is not a non-overlapping single-patch convolution')
 width=projection.weight.numel()//projection.out_channels
 x=hidden_states.reshape(-1,width).to(dtype=projection.weight.dtype)
 return F.linear(x,projection.weight.reshape(projection.out_channels,-1),projection.bias)
def install_patch_embedding(model):
 modules=[m for m in model.modules() if m.__class__.__name__=='Qwen3VLVisionPatchEmbed']
 if len(modules)!=1:raise ValueError('Expected one Qwen3-VL patch embedding module')
 module=modules[0]
 def forward(self,hidden_states):return linear_patch_embed(hidden_states,self.proj)
 module.forward=types.MethodType(forward,module)
 return 'Conv3d per flattened patch replaced by equivalent F.linear; weights unchanged; instance-only override'
