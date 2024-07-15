import math
from numba import cuda

import torch
from torch import Tensor
from torch import nn


@cuda.jit
def relu_kernel(input, output, dim: int):
    """
    Applies ReLU activation to a CUDA array.

    Args:
        input: The input CUDA array.
        output: The output CUDA array.
        dim (int): The total number of elements in the input and output arrays.
    """
    idx = cuda.grid(1)

    if idx < dim:
        output[idx] = max(input[idx], 0)


class NumbaReLU(nn.Module):
    """
    Applies the ReLU function to a CUDA tensor using Numba.

    Args:
        inplace (bool, optional): If set to `True`, the operation will be performed in-place. Defaults to `False`.

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional dimensions
        - Output: :math:`(N, *)`, same shape as the input

    Examples:
        >>> m = NumbaReLU()
        >>> input = torch.randn(2, 3, 4, 5, device='cuda')
        >>> output = m(input)
    """
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.inplace = inplace

    def forward(self, x):
        assert x.is_cuda, "Input must be a CUDA tensor"

        detached_x = x.detach().view(-1)

        output = torch.zeros(x.shape, device=x.device).view(-1)

        threads_per_block = 256
        dim = torch.prod(output.shape).item()
        blocks_per_grid = math.ceil(dim / threads_per_block)

        relu_kernel[blocks_per_grid, threads_per_block](x.detach(), output, dim)

        output = output.view(x.shape)
        return output
    

if __name__ == '__main__':
    numba_relu = NumbaReLU().cuda()
    torch_relu = nn.ReLU().cuda()
    
    input_tensor1 = torch.randint(0, 256, (16, 3, 256, 256), dtype=torch.float32).cuda()

    numba_output = numba_relu(input_tensor1)
    torch_output = torch_relu(input_tensor1)

    print(
        ((numba_output - torch_output) < 1e-6).all(),
    )