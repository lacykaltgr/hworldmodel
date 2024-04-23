import torch
from torch import nn
from torch.distributions import Normal
import numpy as np
from .utils import (
    fft_smooth,
    get_transform, 
    get_constraint,
    get_lr_scheduler,
    get_optimizer
)


class MPC(nn.Module):
    """
    Class for generating more complex optimized inputs
    """
    def __init__(self, config, operation, shape=(1, 28, 28),  device='cpu'):
        self.device = device
        self.shape = shape
        self.operation = operation
        self.log_interval = config.mpc.log_interval
        self.optimizer_class = config.mpc.optimizer_class
        self.optimizer_kwargs = config.mpc.optimizer_kwargs
        self.scheduler_class = config.mpc.scheduler
        self.scheduler_kwargs = config.mpc.scheduler_kwargs
        
        self.iter_n = config.mpc.iter_n         # number of iterations
        self.precond = config.mpc.precond       # preconditioner for the optimization
        self.grad_clip = config.mpc.grad_clip # clip gradients
        self.grad_norm = config.mpc.grad_norm # normalize gradients
        self.bias = config.mpc.bias             # bias for the image
        self.scale = config.mpc.scale           # scale for the image
        
        constraints = nn.Sequential()
        for constraint in config.mpc.constraints:
            constraint_cls = get_constraint(constraint.name)
            constraints.append(constraint_cls(**constraint.kwargs))
        self.constraints = constraints
        
        transforms = nn.Sequential()
        for transform in config.mpc.transforms:
            transform_cls = get_transform(transform.name)
            transforms.append(transform_cls(**transform.kwargs))
        self.transforms = transforms
        
        pre_opt_transforms = nn.Sequential()
        for transform in config.mpc.pre_opt_transforms:
            transform_cls = get_transform(transform.name)
            pre_opt_transforms.append(transform_cls(**transform.kwargs))
        self.pre_opt_transforms = pre_opt_transforms
        
        post_opt_transforms = nn.Sequential()
        for transform in config.mpc.post_opt_transforms:
            transform_cls = get_transform(transform.name)
            post_opt_transforms.append(transform_cls(**transform.kwargs))
        self.post_opt_transforms = post_opt_transforms
        

    def run(self, n_samples, init=None):
        proposal = Proposal(
            shape=self.shape,
            n_samples=n_samples,
            transforms=self.transforms,
            init=init,
            bias=self.bias,
            scale=self.scale,
        )
        optimizer, scheduler = self.make_optimizer(proposal)
        
        results = None
        for i in range(self.iter_n):

            step_results = self.optmizer_step(
                proposal, step_i=i, optimizer=optimizer, scheduler=scheduler
            )

            if i % self.log_interval == 0:
                if results is None:
                    results = step_results
                else:
                    results = {key: torch.cat(
                        [results[key], step_results[key]], dim=0) for key in results.keys()
                               }
        return results


    
    def optimizer_step(self, proposal, step_i, optimizer, scheduler=None):
        """
        Update meitorch in place making a gradient ascent step in the output of net.

        :param process: MEI_result process object
        :param operation: The operation to optimize for
        :param step_i: The step index
        :param add_loss: An additional term to add to the network activation before
                            calling backward on it. Usually, some regularization.
        """
        step_dict = dict()
        optimizer.zero_grad()
        
        inputs = proposal()
        step_dict["proposal"] = inputs.detach()
        
        add_contraint_term = 0
        for additional_constraint in self.constraints:
            add_contraint_term += additional_constraint(inputs)
        
        for pre_opt_transfrom in self.pre_opt_transforms:
            inputs = pre_opt_transfrom(inputs)
            
        step_dict["pre_opt_proposal"] = inputs.detach()

        outputs_dict = self.operation(inputs)
        loss = outputs_dict["loss"].mean()

        # TODO: only accept gradients if loss is ...

        loss += add_contraint_term
        loss.backward()
        
        # TODO: add gradient clipping/ gradient normalization
        if self.grad_clip:
            pass
    
        if self.grad_norm:
            pass
        
        if self.precond:
            for param in proposal.parameters():
                if param.requires_grad and param.grad is not None and len(param.grad.size()) >= 2:
                    smooth_grad = fft_smooth(param.grad, self.precond(step_i))
                    if len(smooth_grad.size()) != len(param.grad.data.size()):
                        smooth_grad = smooth_grad.squeeze(0)
                    param.grad.data.copy_(smooth_grad)

        optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        for post_opt_transform in self.post_opt_transforms:
            inputs = post_opt_transform(inputs)
            
        for key in outputs_dict:
            step_dict[key] = outputs_dict[key].detach()
        return step_dict
    
    def make_optimizer(self, proposal):
        optimizer = get_optimizer(
            params=proposal.parameters(),
            optimizer_type=self.optimizer_class,
            optimizer_kwargs=self.optimizer_kwargs,
            iter_n=self.iter_n
        )
        scheduler = get_lr_scheduler(
            optimizer=self.optimizer,
            scheduler_type=self.scheduler_class,
            scheduler_kwargs=self.scheduler_kwargs
        )
        return optimizer, scheduler
        


class Proposal(nn.Module):
    def __init__(
            self, 
            shape, 
            n_samples=1, 
            transforms=None,
            init=None,  # initial tensor
            bias=0.0,
            scale=1.0,
            device='cpu', 
        ):
        super(Proposal, self).__init__()
        self.device = device
        self.shape = shape
        self.n_samples = n_samples
        self.batch_shape = (n_samples, *shape)
        

        if init is None:
            # normal distribution random intiiation
            # TODO: scaling down scale seemed to work well
            sampler = Normal(
                loc=torch.tensor(bias, device=device), 
                scale=torch.tensor(scale, device=device) #/ 16
            )
            init_samples = sampler.sample(self.batch_shape)
        
        else:
            init_samples = init.to(device)
            if len(init_samples) != n_samples:
                raise ValueError("Number of samples does not match the initial tensor")
            
        self.param = torch.nn.Parameter(
            init_samples, dtype=torch.float32, device=self.device,
            requires_grad=True
        )
        self.transforms = transforms
        
    
    def forward(self):
        samples = self.param # + stop gradient if needed
        for transform in self.transforms:
            samples = transform(samples)
        return samples  

    def detached_param(self):
        return self.param.detach().data

    def __repr__(self):
        return f"Proposal(shape={self.shape}, n_samples={self.n_samples})"

