from typing import List
import torch
from torch.nn import Module


def loss(name: str, keys: List[str] = None):
  def decorator(cls):
    class MaintainedLossWrapper(Module):
      def __init__(self, *args, **kwargs):
        super(MaintainedLossWrapper, self).__init__()
        self.loss_module = cls(*args, **kwargs)
        self.loss_name: str = name
        self.loss_keys: List[str] = keys
        if self.loss_keys is None:
           self.loss_keys = [] 
        self.optimizer = None
        self.grad_scaler = None
        self.grad_clipper = None
        self.opt_params = None
        
      def forward(self, *args, **kwargs):
        return self.loss_module(*args, **kwargs)

      def get_loss_keys(self):
        return self.loss_keys
    
      def get_loss_name(self):
        return self.loss_name
    
      def calculate_loss(self, loss_dict):
          if not self.loss_keys:
            # empty list
            LOSS = loss_dict[self.loss_name]
          else:
            LOSS = 0.
            for key in self.loss_keys:
              LOSS += loss_dict[key]
          return LOSS
    
      def with_optimizer(self, params, optimizer_cls=None, **optimizer_kwargs):
        if optimizer_cls is None:
          optimizer_cls = torch.optim.Adam
        
        self.optimizer = optimizer_cls(params, **optimizer_kwargs)
        self.grad_scaler = torch.cuda.amp.GradScaler()
        self.grad_clipper = lambda max_norm: torch.nn.utils.clip_grad_norm_(params, max_norm)
        return self
    
      def clip_grads(self, max_norm):
        self.grad_clipper(max_norm)
    
    return MaintainedLossWrapper

  return decorator