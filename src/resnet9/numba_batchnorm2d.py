import math
from numba import cuda
import torch
from torch import Tensor
from torch import nn


# TODO: Parallelize mean and var function
@cuda.jit
def batchnorm2d_kernel(input, output, mean, var, eps, gamma, beta):
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
        output[batch_idx, channel, out_h, out_w] = (input[batch_idx, channel, out_h, out_w] - mean[channel]) / math.sqrt(var[channel] + eps)
        
        if gamma is not None and beta is not None:
            output[batch_idx, channel, out_h, out_w] = output[batch_idx, channel, out_h, out_w] * gamma[channel] + beta[channel]


class NumbaBatchNorm2d(nn.Module):
    """
    A PyTorch module that implements a batch normalization layer using Numba for acceleration.

    This class is similar to `torch.nn.BatchNorm2d` but uses Numba to perform the mean and variance
    calculations on the GPU, potentially leading to faster execution.

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
    """
    def __init__(self,
                 num_features: int,
                 eps: float = 1e-05,
                 momentum: float = 0.1,
                 affine: bool = True,
                 track_running_stats: bool = True,) -> None:
        super().__init__()

        self.eps = eps
        self.momentum = momentum
        self.track_running_stats = track_running_stats

        if affine:
            self.gamma = nn.Parameter(data=torch.ones(num_features))
            self.beta = nn.Parameter(data=torch.zeros(num_features))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

        if self.track_running_stats:
            self.running_mean = 0
            self.running_var = 1
        
    def forward(self, x: Tensor):
        assert x.is_cuda, "Input must be a CUDA tensor"
        assert x.dim() == 4, "Input must be a 4D tensor"

        if self.training:
            mean = torch.mean(x, dim=(0, 2, 3)) # calculate mean over batch
            var = torch.var(x, dim=(0, 2, 3), unbiased=False) # calculate variance over batch

            # update running estimations
            if self.track_running_stats:
                self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
                self.running_var = (1- self.momentum) * self.running_var + self.momentum * x.var(dim=(0, 2, 3), unbiased=True)

        else:
            mean = self.running_mean
            var = self.running_var

        output = torch.zeros(x.shape, device=x.device)
        
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            math.ceil(x.shape[0] * x.shape[1] / threads_per_block[0]),
            math.ceil(x.shape[2] / threads_per_block[1]),
            math.ceil(x.shape[3] / threads_per_block[2])
        )

        batchnorm2d_kernel[blocks_per_grid, threads_per_block](
            x.detach(), output, mean, var, self.eps, self.gamma.detach(), self.beta.detach()
        )

        return output
    

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