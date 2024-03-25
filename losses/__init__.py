from typing import List
import torch
from torch.nn import ModuleList


def loss(name: str, keys: List[str] = None):
  def decorator(cls):
      
    class LoggedLossModule(cls):
      def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_name: str = name
        self.loss_keys: List[str] = keys
        if self.loss_keys is None:
           self.loss_keys = [] 
        self.optimizer = None
        self.grad_scaler = None
        self.parameters = None

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
    
      def optimize(self, modules, optimizer_cls=None, **optimizer_kwargs):
        if optimizer_cls is None:
          optimizer_cls = torch.optim.Adam
        self.parameters = ModuleList(modules).parameters()
        self.optimizer = optimizer_cls(self.parameters, **optimizer_kwargs)
        self.grad_scaler = torch.cuda.amp.GradScaler()
        return self
    
      def clip_grads(self, max_norm):
        torch.nn.utils.clip_grad_norm_(self.parameters, max_norm)
    
    return LoggedLossModule

  return decorator