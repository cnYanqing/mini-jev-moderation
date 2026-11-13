"""Two fixed questions, four independently selected heads, one shared Mini backbone."""
import torch
from jev.model import AnswerHead
class RoutedHead(torch.nn.Module):
 def __init__(self,states,hidden,dim):
  super().__init__();self.heads=torch.nn.ModuleDict();self.modality=None
  for key,state in states.items():
   head=AnswerHead(hidden,dim);head.load_state_dict(state);self.heads[key]=head
 def forward(self,hs,last,spans,types=None,calibrated=True):
  if self.modality not in ('text','image') or len(hs)!=2:raise RuntimeError('Specialist requires the fixed sexual/violence pair in one cached batch')
  out=[]
  for i,label in enumerate(('sexual','violence')):
   head=self.heads[self.modality+'_'+label]
   out.extend(head(hs[i:i+1],[last[i]],[spans[i]],[types[i]] if types else None,calibrated))
  return out
