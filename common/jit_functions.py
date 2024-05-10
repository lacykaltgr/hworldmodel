import torch

@torch.jit.script
def symlog(x):
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)

@torch.jit.script
def symexp(x):
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1.0)


@torch.jit.script
def hyperbolic_forward(x):
    epsilon = 1e-3
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + epsilon * x


@torch.jit.script
def hyperbolic_backward(x):
    epsilon = 1e-3
    return torch.sign(x) * (torch.square(
        torch.sqrt(1 + 4 * epsilon * (epsilon + 1 + torch.abs(x))) / 2 / epsilon -
        1 / 2 / epsilon) - 1)
