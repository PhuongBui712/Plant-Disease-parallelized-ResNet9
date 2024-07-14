import math
from numba import cuda, float32

import torch
from torch import nn

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
    # Define an array in the shared memory
    # The size and type of the arrays must be known at compile time
    sA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    sB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    x, y = cuda.grid(2)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bpg = cuda.gridDim.x    # blocks per grid

    # Each thread computes one element in the result matrix.
    # The dot product is chunked into dot products of TPB-long vectors.
    tmp = float32(0.)
    for i in range(bpg):
        # Preload data into shared memory
        sA[ty, tx] = 0
        sB[ty, tx] = 0
        if y < input.shape[0] and (tx+i*TPB) < input.shape[1]:
          sA[ty, tx] = input[y, tx + i * TPB]
        if x < weight.shape[1] and (ty+i*TPB) < weight.shape[0]:
          sB[ty, tx] = weight[ty + i * TPB, x]

        # Wait until all threads finish preloading
        cuda.syncthreads()

        # Computes partial product on the shared memory
        for j in range(TPB):
            tmp += sA[ty, j] * sB[j, tx]

        # Wait until all threads finish computing
        cuda.syncthreads()
    if y < output.shape[0] and x < output.shape[1]:
        output[y, x] = tmp
        

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
                 bias: bool = True,
                 custom_weight = None,
                 custom_bias = None) -> None:
        super().__init__()

        bound = math.sqrt(1.0 / in_features)
        self.weight = nn.Parameter(torch.rand(size=(out_features, in_features)) * 2 * bound - bound)
        if bias:
            self.bias = nn.Parameter(torch.rand(out_features) * 2 * bound - bound)
        else:
            self.register_parameter('bias', None)
            
        if custom_weight is not None:
            self.weight = custom_weight
        if custom_bias is not None:
            self.bias = custom_bias

    def forward(self, x):
        assert x.is_cuda, "Input must be a CUDA tensor"
        assert self.weight.is_cuda, "Weights must be CUDA tensors"
        assert self.bias is None or self.bias.is_cuda, "Bias must be a CUDA tensor if it exists"

        original_shape = x.shape
        detached_x = x.detach()
        if x.dim() > 2:
            detached_x = detached_x.flatten(0, -2)

        output = torch.empty(detached_x.size(0), self.weight.shape[0], device=x.device)
        
        threads_per_block = (TPB, TPB)
        grid_y_max = max(detached_x.shape[0], self.weight.shape[0])
        grid_x_max = max(detached_x.shape[1], self.weight.shape[1])

        blocks_per_grid_x = math.ceil(grid_x_max / threads_per_block[0])
        blocks_per_grid_y = math.ceil(grid_y_max / threads_per_block[1])

        blocks_per_grid = (blocks_per_grid_x, blocks_per_grid_y)
        linear_kernel[blocks_per_grid, threads_per_block](
            detached_x, output, self.weight.detach().T
        )

        if self.bias is not None:
            output += self.bias
        
        output = output.view(*original_shape[:-1], output.shape[-1])
        return output
    

if __name__ == '__main__':
    input_tensor1 = torch.randint(0, 256, (16, 3, 256, 256), dtype=torch.float32).cuda()

    torch_linear = nn.Linear(256, 512, dtype=torch.float32).cuda()
    numba_linear = NumbaLinear(256, 512, custom_weight=torch_linear.weight, custom_bias=torch_linear.bias).cuda()

    numba_output = numba_linear(input_tensor1)
    torch_output = torch_linear(input_tensor1)

    print(((numba_output - torch_output) < 1e-6).all())