import math
from numba import cuda
from typing import Optional, Tuple
import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from torch.autograd import Function


@cuda.jit
def batchnorm2d_forward_kernel(input, output, mean, inv_std, gamma, beta):
    """
    A CUDA kernel that performs batch normalization on a 4D tensor.

    Args:
        input: The input tensor.
        output: The output tensor.
        mean: The mean of the input tensor.
        var: The variance of the input tensor.
        eps (float): A small value added to the denominator for numerical stability.
        gamma: The scaling factor.
        beta: The shifting factor.
    """
    idx, out_h, out_w = cuda.grid(3)

    batch_idx = idx // input.shape[1]
    channel = idx % input.shape[1]

    if batch_idx < output.shape[0] and channel < output.shape[1] and out_h < output.shape[2] and out_w < output.shape[3]:
        normalized = (input[batch_idx, channel, out_h, out_w] - mean[channel]) * inv_std[channel]
        output[batch_idx, channel, out_h, out_w] = normalized * gamma[channel] + beta[channel]


class NumbaBatchNorm2dFunction(Function):
    @staticmethod
    def forward(ctx,
                input: Tensor,
                gamma: Tensor, 
                beta: Tensor, 
                running_mean: Optional[Tensor], 
                running_var: Optional[Tensor], 
                eps: float, 
                momentum: float, 
                training: bool) -> Tensor:
        input = input.contiguous()
        
        if training:
            mean = input.mean(dim=(0, 2, 3))
            var = input.var(dim=(0, 2, 3), unbiased=False)
            
            if running_mean is not None:
                running_mean.mul_(1 - momentum).add_(mean * momentum)
            if running_var is not None:
                running_var.mul_(1 - momentum).add_(var * momentum)
        else:
            mean = running_mean
            var = running_var
        
        inv_std = 1 / torch.sqrt(var + eps)
        output = torch.empty_like(input)
        
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            math.ceil(input.shape[0] * input.shape[1] / threads_per_block[0]),
            math.ceil(input.shape[2] / threads_per_block[1]),
            math.ceil(input.shape[3] / threads_per_block[2])
        )

        batchnorm2d_forward_kernel[blocks_per_grid, threads_per_block](
            input.detach(), output, mean.detach(), inv_std.detach(), gamma.detach(), beta.detach()
        )
        
        ctx.save_for_backward(input, gamma, mean, inv_std)
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None, None, None, None, None]:
        input, gamma, mean, inv_std = ctx.saved_tensors
        
        # Use PyTorch's built-in backward pass for simplicity and correctness
        normalized = (input - mean[None, :, None, None]) * inv_std[None, :, None, None]
        grad_input = F.batch_norm(
            input, mean, 1/inv_std**2, gamma, None, 
            eps=0, momentum=0, training=True
        )
        grad_input = grad_output * grad_input
        
        grad_gamma = (grad_output * normalized).sum(dim=(0, 2, 3))
        grad_beta = grad_output.sum(dim=(0, 2, 3))
        
        return grad_input, grad_gamma, grad_beta, None, None, None, None, None


class NumbaBatchNorm2d(nn.Module):
    """
    A PyTorch module that implements a 2D batch normalization layer using Numba for acceleration.

    This class is a drop-in replacement for `torch.nn.BatchNorm2d`, but utilizes Numba to perform the
    mean and variance calculations on the GPU, potentially leading to faster execution. It maintains
    the same functionality and arguments as the standard `BatchNorm2d` layer, including support for
    affine transformations, running statistics tracking, and training/evaluation modes.

    Args:
        num_features (int): The number of features in the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability.
            Defaults to 1e-05.
        momentum (float, optional): The momentum used for running mean and variance computation.
            Defaults to 0.1.
        affine (bool, optional): If True, the layer will learn affine parameters (gamma and beta).
            Defaults to True.
        track_running_stats (bool, optional): If True, the layer will track running mean and variance.
            Defaults to True.

    Example:
        >>> batchnorm = NumbaBatchNorm2d(num_features=64).cuda()
        >>> input_tensor = torch.randn(16, 64, 32, 32, device='cuda')
        >>> output_tensor = batchnorm(input_tensor)
    """
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True) -> None:
                 
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)
        
        if self.track_running_stats:
            self.register_buffer('running_mean', torch.zeros(num_features))
            self.register_buffer('running_var', torch.ones(num_features))
        else:
            self.register_buffer('running_mean', None)
            self.register_buffer('running_var', None)

    def forward(self, x: Tensor):
        return NumbaBatchNorm2dFunction.apply(
            x, self.weight, self.bias, 
            self.running_mean, self.running_var, 
            self.eps, self.momentum, self.training
        )


if __name__ == '__main__':
    numba_batchnorm = NumbaBatchNorm2d(3).cuda()
    torch_batchnorm = nn.BatchNorm2d(3).cuda()

    input_tensor1 = torch.randint(0, 256, (16, 3, 256, 256), dtype=torch.float32).cuda()
    input_tensor2 = torch.randint(0, 256, (16, 3, 256, 256), dtype=torch.float32).cuda()

    numba_output = numba_batchnorm(input_tensor1)
    numba_output = numba_batchnorm(input_tensor2)
    torch_output = torch_batchnorm(input_tensor1)
    torch_output = torch_batchnorm(input_tensor2)

    print(
        ((numba_output - torch_output) < 1e-6).all(),
        ((torch_batchnorm.running_mean - numba_batchnorm.running_mean) < 1e-6).all(),
        ((torch_batchnorm.running_var - numba_batchnorm.running_var) < 1e-6).all()
    )