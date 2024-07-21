import math
from numba import cuda, float32
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.autograd import Function

TPB = 32

@cuda.jit
def linear_kernel(input, output, weight):
    """
    Performs a matrix multiplication between an input matrix and a weight matrix using shared memory.

    Args:
        input: The input matrix.
        output: The output matrix.
        weight: The weight matrix.
    """
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x

    tmp = 0.0
    for i in range(bpg):
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < input.shape[0] and (tx+i*TPB) < input.shape[1]:
            sA[ty, tx] = input[y, tx + i * TPB]
        if x < weight.shape[1] and (ty+i*TPB) < weight.shape[0]:
            sB[ty, tx] = weight[ty + i * TPB, x]

        cuda.syncthreads()

        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        cuda.syncthreads()
    if y < output.shape[0] and x < output.shape[1]:
        output[y, x] = tmp


class NumbaLinearFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
        ctx.save_for_backward(input, weight, bias)
        
        output = torch.empty(input.size(0), weight.size(0), device=input.device)
        
        threads_per_block = (TPB, TPB)
        grid_y_max = max(input.shape[0], weight.shape[0])
        grid_x_max = max(input.shape[1], weight.shape[1])

        blocks_per_grid_x = math.ceil(grid_x_max / threads_per_block[0])
        blocks_per_grid_y = math.ceil(grid_y_max / threads_per_block[1])

        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        
        linear_kernel[blocks_per_grid, threads_per_block](
            input.detach(), output, weight.detach().T
        )
        
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        
        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
        input, weight, bias = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        if ctx.needs_input_grad[1]:
            grad_weight = grad_output.t().mm(input)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight, grad_bias


class NumbaLinear(nn.Module):
    """
    Performs a linear transformation on a tensor using Numba CUDA.

    This class implements a linear transformation with configurable input and output features, and optional bias.
    It leverages Numba CUDA for efficient GPU acceleration.

    Args:
        in_features (int): The number of input features.
        out_features (int): The number of output features.
        bias (bool, optional): Whether to use a bias term. Defaults to True.
        custom_weight (torch.Tensor, optional): A custom weight tensor to use. Defaults to None.
        custom_bias (torch.Tensor, optional): A custom bias tensor to use. Defaults to None.
    """
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: Tensor):
        return NumbaLinearFunction.apply(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


if __name__ == '__main__':
    input_tensor1 = torch.randint(0, 256, (16, 3, 256, 256), dtype=torch.float32).cuda()

    torch_linear = nn.Linear(256, 512, dtype=torch.float32).cuda()
    numba_linear = NumbaLinear(256, 512, custom_weight=torch_linear.weight, custom_bias=torch_linear.bias).cuda()

    numba_output = numba_linear(input_tensor1)
    torch_output = torch_linear(input_tensor1)

    print(((numba_output - torch_output) < 1e-6).all())