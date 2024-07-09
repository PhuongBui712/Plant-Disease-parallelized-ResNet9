import math
from numba import cuda
from typing import Optional

import torch
from torch import nn


MIN_FLOAT32 = torch.finfo(torch.float32).min

@cuda.jit
def max_pool_2d_kernel(input: cuda.devicearray.DeviceNDArray,
                       output: cuda.devicearray.DeviceNDArray,
                       kernel_size: int,
                       padding: int,
                       stride: int):
    """
    Performs a 2D max pooling operation on a 4D tensor.

    Args:
        input (cuda.devicearray.DeviceNDArray): The input tensor.
        output (cuda.devicearray.DeviceNDArray): The output tensor.
        kernel_size (int): The size of the pooling kernel.
        padding (int): The amount of padding to apply.
        stride (int): The stride of the pooling operation.
    """
    idx, out_h, out_w = cuda.grid(3)
    
    batch_idx = idx // input.shape[1]
    channel = idx % input.shape[1]
    
    if batch_idx < input.shape[0] and channel < input.shape[1] and out_h < input.shape[2] and out_w < input.shape[3]:
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                in_y = out_h * stride - padding + ky
                in_x = out_w * stride - padding +kx

                if 0 <= in_y < input.shape[2] and 0 <= in_x < input.shape[3]:
                    output[batch_idx, channel, out_h, out_w] = max(output[batch_idx, channel, out_h, out_w],
                                                                   input[batch_idx, channel, in_y, in_x])


class NumbaMaxPool2d(nn.Module):
    """
    Performs a 2D max pooling operation on a 4D tensor using Numba CUDA.

    This class implements a max pooling operation with configurable kernel size, padding, and stride.
    It leverages Numba CUDA for efficient GPU acceleration.

    Args:
        kernel_size (int): The size of the pooling kernel.
        padding (Optional[int], optional): The amount of padding to apply. Defaults to 0.
        stride (Optional[int], optional): The stride of the pooling operation. Defaults to 1.

    Example:
        >>> pool = NumbaMaxPool2d(kernel_size=2, padding=1, stride=2)
        >>> input_tensor = torch.randn(16, 3, 512, 512, device='cuda')
        >>> output_tensor = pool(input_tensor)
    """
    def __init__(self,
                 kernel_size: int,
                 padding: Optional[int] = 0,
                 stride: Optional[int] = 1):
        super().__init__()

        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, x):
        assert x.is_cuda, "Input must be a CUDA tensor"
        assert x.dim() == 4, "Input must be a 4D tensor"

        detached_x = x.detach()

        batch_size, channels, in_height, in_width = x.shape
        out_height = (in_height + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1
        out_width = (in_width + 2 * self.padding - (self.kernel_size - 1) - 1) // self.stride + 1

        output = torch.full(
            size=(batch_size, channels, out_height, out_width),
            fill_value=MIN_FLOAT32,
            device=x.device
        )
        
        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            math.ceil(batch_size * channels / threads_per_block[0]),
            math.ceil(out_height / threads_per_block[1]),
            math.ceil(out_width / threads_per_block[2])
        )

        max_pool_2d_kernel[blocks_per_grid, threads_per_block](
            detached_x, output, self.kernel_size, self.padding, self.stride
        )

        return output
    

@cuda.jit
def _maxpool2d_kernel_2(input, output, kernel_size, stride, in_height, in_width, out_height, out_width, min_val):
    # Calculate indices
    idx = cuda.threadIdx.x + cuda.blockIdx.x * cuda.blockDim.x
    idy = cuda.threadIdx.y + cuda.blockIdx.y * cuda.blockDim.y
    idz = cuda.threadIdx.z + cuda.blockIdx.z * cuda.blockDim.z

    # Map to 4D indices
    batch = idx // input.shape[1]
    channel = idx % input.shape[1]
    x = idy
    y = idz

    if batch < input.shape[0] and channel < input.shape[1] and x < out_height and y < out_width:
        max_val = min_val
        for i in range(kernel_size):
            for j in range(kernel_size):
                in_x = x * stride + i
                in_y = y * stride + j
                if in_x < in_height and in_y < in_width:
                    val = input[batch, channel, in_x, in_y]
                    if val > max_val:
                        max_val = val
        output[batch, channel, x, y] = max_val

class NumbaMaxPool2d_2(torch.nn.Module):
    def __init__(self, kernel_size, stride=None):
        super(NumbaMaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size

    def forward(self, x):
        if not x.is_cuda:
            x = x.cuda()
        
        # Ensure input is float32
        x = x.float()
        
        input_shape = x.shape
        output_shape = (
            input_shape[0],  # batch size
            input_shape[1],  # channels
            (input_shape[2] - self.kernel_size) // self.stride + 1,  # height
            (input_shape[3] - self.kernel_size) // self.stride + 1   # width
        )
        
        output = torch.cuda.FloatTensor(*output_shape).fill_(MIN_FLOAT32)
        
        threads_per_block = (64, 64, 64)
        blocks_per_grid = (
            (input_shape[0] * input_shape[1] + threads_per_block[0] - 1) // threads_per_block[0],
            (output_shape[2] + threads_per_block[1] - 1) // threads_per_block[1],
            (output_shape[3] + threads_per_block[2] - 1) // threads_per_block[2]
        )
        
        _maxpool2d_kernel_2[blocks_per_grid, threads_per_block](
            x,
            output,
            self.kernel_size,
            self.stride,
            input_shape[2],  # in_height
            input_shape[3],  # in_width
            output_shape[2],  # out_height
            output_shape[3],  # out_width
            MIN_FLOAT32  # Pass the minimum value as an argument
        )
        
        return output
    

if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256, device='cuda')

    torch_max_pooling = nn.MaxPool2d(kernel_size=3, stride=2).cuda()
    torch_output = torch_max_pooling(input)

    numba_max_pooling = NumbaMaxPool2d(kernel_size=3, stride=2).cuda()
    numba_output = numba_max_pooling(input)

    assert ((torch_output - numba_output) < 1e-6).all().item(), "Numba MaxPool2D output does not match PyTorch MaxPool2D output"
