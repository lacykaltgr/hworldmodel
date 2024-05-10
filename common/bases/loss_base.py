from typing import List
import torch
from torch import nn
from warnings import warn


"""
Loss wrapper can be used two ways:

1. As a simple wrapper class

    loss_module = MyLossModule()
    loss = LossWrapper(
        loss_module=loss_module,
    )
    
2. As a decorator

    @loss
    class MyLossModule(nn.Module):
        def __init__(self, *args, **kwargs):
            super(MyLossModule, self).__init__()
            # ...

"""


class LossWrapper(nn.Module):
    def __init__(
            self,
            loss_module,    # LossModule
            
            optimizer_cls=None,
            optimizer_kwargs=None,
            use_grad_scaler=True,   # bool
        ):
        super(LossWrapper, self).__init__()
        self.loss_module = loss_module
        
        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.use_grad_scaler = use_grad_scaler
        
        self.optimizer = None
        self.grad_scaler = None
        self.grad_clip_fn = None
    
    def forward(self, *args, **kwargs):
        return self.loss_module(*args, **kwargs)
    

    def calculate_loss(self, loss_dict):
        LOSS = 0
        for key in loss_dict.keys():
            LOSS += loss_dict[key]
        #if not self.loss_keys:
        # empty list
        #    LOSS = loss_dict[self.loss_name]
        #else:
        #    LOSS = 0.
        #    for key in self.loss_keys:
        #        LOSS += loss_dict[key]
        return LOSS

    def with_optimizer(
        self, 
        opt_module=None, 
        optimizer_cls=None, 
        optimizer_kwargs=None,
        use_grad_scaler=None,
    ):
        """
        Uses torch.optim.Adam as default optimizer
        """
        if self.optimizer is not None:
            warn("Optimizer already initialized. Reinitializing.")
            
        if optimizer_cls is None:
            optimizer_cls = self.optimizer_cls
        else:
            self.optimizer_cls = optimizer_cls
            
        if optimizer_kwargs is None:
            optimizer_kwargs = self.optimizer_kwargs
        else:
            self.optimizer_kwargs = optimizer_kwargs
    
        if opt_module is None:
            opt_module = self.loss_module
        else:
            self.opt_module = opt_module
        self.optimizer = optimizer_cls(opt_module.parameters(), **optimizer_kwargs)
        
        
        if use_grad_scaler is None:
            use_grad_scaler = self.use_grad_scaler
        else:
            self.use_grad_scaler = use_grad_scaler
        if use_grad_scaler:
            self.grad_scaler = torch.cuda.amp.GradScaler()
            
        return self

    def clip_grads(self, max_norm):
        grad = torch.nn.utils.clip_grad_norm_(self.opt_module.parameters(), max_norm)
        return grad


def loss(cls):
    class DecoratorLossWrapper(LossWrapper):
        def __init__(self, *args, **kwargs):
            super(DecoratorLossWrapper, self).__init__(
                loss_module=cls(*args, **kwargs)
            )
    return DecoratorLossWrapper