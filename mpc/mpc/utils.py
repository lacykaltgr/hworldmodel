import torch
import numpy as np
from torch import optim

from .optimizer import MEIoptimizer, MEIBatchoptimizer
from . import transforms
from . import constraints


def batch_mean(batch, keepdim=False):
    """ Compute mean for a batch of images. """
    mean = batch.view(len(batch), -1).mean(-1)
    if keepdim:
        mean = mean.view(len(batch), 1, 1, 1)
    return mean


def batch_std(batch, keepdim=False, unbiased=True):
    """ Compute std for a batch of images. """
    std = batch.view(len(batch), -1).std(-1, unbiased=unbiased)
    if keepdim:
        std = std.view(len(batch), 1, 1, 1)
    return std

def fft_smooth(grad, factor=1/4):
    """
    Tones down the gradient with 1/sqrt(f) filter in the Fourier domain.
    Equivalent to low-pass filtering in the spatial domain.



    :param grad: The gradient
    :param factor: The factor
    """
    if factor == 0:
        return grad
    h, w = grad.size()[-2:]
    tw = np.minimum(np.arange(0, w), np.arange(w, 0, -1), dtype=np.float32)
    th = np.minimum(np.arange(0, h), np.arange(h, 0, -1), dtype=np.float32)
    t = 1 / np.maximum(1.0, (tw[None, :] ** 2 + th[:, None] ** 2) ** factor)
    F = grad.new_tensor(t / t.mean()).unsqueeze(0)
    pp = torch.fft.fft2(grad.data, dim=(-1, -2))
    return torch.real(torch.fft.ifft2(pp * F, dim=(-1, -2)))


def roll(tensor, shift, axis):
    """
    Rolls the tensor along the given axis by the given shift

    :param tensor: The tensor to roll
    :param shift: The shift to apply
    :param axis: The axis on which to shift
    :return: The shifted tensor
    """
    if shift == 0:
        return tensor

    if axis < 0:
        axis += tensor.dim()

    dim_size = tensor.size(axis)
    after_start = dim_size - shift
    if shift < 0:
        after_start = -shift
        shift = dim_size - abs(shift)

    before = tensor.narrow(axis, 0, dim_size - shift)
    after = tensor.narrow(axis, after_start, shift)
    return torch.cat([after, before], axis)


def get_transform(name):
    if hasattr(transforms, name):
        return getattr(transforms, name)
    raise ValueError(f"Transform {name} not found. Check the transforms module.")
    
def get_constraint(name):
    if hasattr(constraints, name):
        return getattr(constraints, name)
    raise ValueError(f"Constraint {name} not found. Check the constraints module.")
    
    
def get_lr_scheduler(optimizer, scheduler_type, kwargs):

    """
    StepLR:     This scheduler decreases the learning rate at specified epochs by a multiplicative factor.
                For example, you could use this scheduler to decrease the learning rate by 0.5 every 10 epochs.

                - step_size (int): The number of epochs between each learning rate decrease.
                - gamma (float): The multiplicative factor by which the learning rate is decreased at each step.

    MultiStepLR: This scheduler is similar to StepLR, but it allows you to specify multiple epochs
                    at which to decrease the learning rate. For example,
                    you could use this scheduler to decrease the learning rate by 0.5 at epochs 10, 20, and 30.

                - milestones (list[int]): A list of epochs at which to decrease the learning rate.
                - gamma (float): The multiplicative factor by which the learning rate is decreased at each step.

    ExponentialLR: This scheduler decreases the learning rate exponentially by a multiplicative factor at each epoch.

                - gamma (float): The multiplicative factor by which the learning rate is decreased at each epoch.

    CosineAnnealingLR: This scheduler decreases the learning rate following a cosine curve,
                        starting from a high value and gradually decreasing to a low value.

                - T_max (int): The maximum number of epochs.
                - eta_min (float): The minimum learning rate.

    CosineAnnealingWarmRestartsLR: This scheduler is similar to CosineAnnealingLR,
                                    but it restarts the cosine annealing process at specified epochs.
                                    This can be useful for preventing the model from getting stuck in local minima.

                - T_max (int): The maximum number of epochs per restart.
                - eta_min (float): The minimum learning rate.
                - T_mult (float): A factor by which the T_max is multiplied after each restart.

    :param optimizer: the optimizer to be used
    :param hparams: the hyperparameters of the schdeler
    :return: the scheduler
    """

    if scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer, **kwargs)
    elif scheduler_type == "multi_step":
        return optim.lr_scheduler.MultiStepLR(optimizer, **kwargs)
    elif scheduler_type == "exponential":
        return optim.lr_scheduler.ExponentialLR(optimizer, **kwargs)
    elif scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer, **kwargs)
    elif scheduler_type == "cosine_warm":
        return optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **kwargs)
    
    
def get_optimizer(
        params,
        optimizer_type: str = "adam",
        iter_n = None,
        **kwargs):
    """
    Get an optimizer based on the optimizer type and the learning rate

    :param params:
    :param optimizer_type: The optimizer type
    :param iter_n: The number of iterations
    :param kwargs: Additional arguments for the optimizer
    :return: The optimizer nn.Optimizer instance
    """
    if optimizer_type == "adam":
        return optim.Adam(params, **kwargs)
    elif optimizer_type == "sgd":
        return optim.SGD(params, **kwargs)
    elif optimizer_type == "rmsprop":
        return optim.RMSprop(params, **kwargs)
    elif optimizer_type == "mei":
        assert iter_n is not None, "iter_n must be specified"
        kwargs.update({"iter_n": iter_n})
        return MEIoptimizer(params, kwargs)
    elif optimizer_type == "meibatch":
        assert iter_n is not None, "iter_n must be specified"
        kwargs.update({"iter_n": iter_n})
        return MEIBatchoptimizer(params, kwargs)
    else:
        raise ValueError("Invalid optimizer type (must be 'adam', 'sgd' or 'rmsprop')")