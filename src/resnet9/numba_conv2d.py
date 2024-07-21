import math
from numba import cuda
from typing import Optional

import torch
from torch import Tensor
from torch import nn


@cuda.jit
def conv2d_kernel_shared(input, kernel, output, padding: int, stride: int):
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    out_height, out_width = output.shape[2:]

    # Shared memory for input and kernel
    shared_input = cuda.shared.array(shape=(18, 18), dtype=float32)
    shared_kernel = cuda.shared.array(shape=(3, 3), dtype=float32)

    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y
    bx = cuda.blockIdx.x
    by = cuda.blockIdx.y
    bz = cuda.blockIdx.z

    out_x = bx * cuda.blockDim.x + tx
    out_y = by * cuda.blockDim.y + ty

    batch_idx = bz // out_channels
    out_channel_idx = bz % out_channels

    if batch_idx < batch_size and out_channel_idx < out_channels and out_y < out_height and out_x < out_width:
        res = 0.0
        for in_channel in range(in_channels):
            # Load input data into shared memory
            for i in range(0, cuda.blockDim.y + kernel_height - 1):
                for j in range(0, cuda.blockDim.x + kernel_width - 1):
                    in_y = by * cuda.blockDim.y + i - padding
                    in_x = bx * cuda.blockDim.x + j - padding
                    if 0 <= in_y < in_height and 0 <= in_x < in_width:
                        shared_input[i, j] = input[batch_idx, in_channel, in_y, in_x]
                    else:
                        shared_input[i, j] = 0.0

            # Load kernel data into shared memory
            if ty < kernel_height and tx < kernel_width:
                shared_kernel[ty, tx] = kernel[out_channel_idx, in_channel, ty, tx]

            cuda.syncthreads()

            # Compute convolution
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    in_y = ty + ky
                    in_x = tx + kx
                    res += shared_input[in_y, in_x] * shared_kernel[ky, kx]

            cuda.syncthreads()

        output[batch_idx, out_channel_idx, out_y, out_x] = res


@cuda.jit
def conv2d_backward_input(grad_output, weight, grad_input, padding, stride):
    combined_idx, in_y, in_x = cuda.grid(3)
    batch_size, in_channels, in_height, in_width = grad_input.shape
    out_channels, _, kernel_height, kernel_width = weight.shape
    out_height, out_width = grad_output.shape[2:]

    batch_idx = combined_idx // in_channels
    in_channel_idx = combined_idx % in_channels

    if batch_idx < batch_size and in_channel_idx < in_channels and in_y < in_height and in_x < in_width:
        grad = 0.0
        for out_channel in range(out_channels):
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    out_y = (in_y + padding - ky) // stride
                    out_x = (in_x + padding - kx) // stride
                    if 0 <= out_y < out_height and 0 <= out_x < out_width:
                        if (in_y + padding - ky) % stride == 0 and (in_x + padding - kx) % stride == 0:
                            grad += grad_output[batch_idx, out_channel, out_y, out_x] * weight[out_channel, in_channel_idx, ky, kx]
        grad_input[batch_idx, in_channel_idx, in_y, in_x] = grad


@cuda.jit
def conv2d_backward_weight(input, grad_output, grad_weight, padding, stride):
    combined_idx, ky, kx = cuda.grid(3)
    out_channels, in_channels, kernel_height, kernel_width = grad_weight.shape
    batch_size, _, in_height, in_width = input.shape
    out_height, out_width = grad_output.shape[2:]

    out_channel_idx = combined_idx // in_channels
    in_channel_idx = combined_idx % in_channels

    if out_channel_idx < out_channels and in_channel_idx < in_channels and ky < kernel_height and kx < kernel_width:
        grad = 0.0
        for batch_idx in range(batch_size):
            for out_y in range(out_height):
                for out_x in range(out_width):
                    in_y = out_y * stride - padding + ky
                    in_x = out_x * stride - padding + kx
                    if 0 <= in_y < in_height and 0 <= in_x < in_width:
                        grad += input[batch_idx, in_channel_idx, in_y, in_x] * grad_output[batch_idx, out_channel_idx, out_y, out_x]
        grad_weight[out_channel_idx, in_channel_idx, ky, kx] = grad


class Conv2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, padding, stride):
        ctx.save_for_backward(input, weight, bias)
        ctx.padding = padding
        ctx.stride = stride

        input_data = input.detach()
        weight_data = weight.detach()

        batch_size, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        out_height = (in_height + 2 * padding - kernel_height) // stride + 1
        out_width = (in_width + 2 * padding - kernel_width) // stride + 1

        output = torch.zeros(batch_size, out_channels, out_height, out_width,
                             dtype=input.dtype, device=input.device)

        threads_per_block = (16, 16)
        blocks_per_grid = (
            math.ceil(out_width / threads_per_block[0]),
            math.ceil(out_height / threads_per_block[1]),
            batch_size * out_channels
        )

        conv2d_kernel_shared[blocks_per_grid, threads_per_block](
            input_data, weight_data, output, padding, stride
        )

        return output + bias.view(1, -1, 1, 1)


    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors
        padding, stride = ctx.padding, ctx.stride

        grad_input = grad_weight = grad_bias = None

        # Detach tensors for CUDA operations
        input_data = input.detach()
        weight_data = weight.detach()
        grad_output_data = grad_output.detach()

        if ctx.needs_input_grad[0]:
            grad_input = torch.zeros_like(input)
            threads_per_block = (8, 8, 8)
            blocks_per_grid = (
                math.ceil(input.shape[0] * input.shape[1] / threads_per_block[0]),
                math.ceil(input.shape[2] / threads_per_block[1]),
                math.ceil(input.shape[3] / threads_per_block[2])
            )
            conv2d_backward_input[blocks_per_grid, threads_per_block](
                grad_output_data, weight_data, grad_input, padding, stride
            )

        if ctx.needs_input_grad[1]:
            grad_weight = torch.zeros_like(weight)
            threads_per_block = (8, 8, 8)
            blocks_per_grid = (
                math.ceil(weight.shape[0] * weight.shape[1] / threads_per_block[0]),
                math.ceil(weight.shape[2] / threads_per_block[1]),
                math.ceil(weight.shape[3] / threads_per_block[2])
            )
            conv2d_backward_weight[blocks_per_grid, threads_per_block](
                input_data, grad_output_data, grad_weight, padding, stride
            )

        if ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None


class NumbaConv2d(torch.nn.Module):
    """
    Performs a 2D convolution operation on a 4D tensor using Numba CUDA.

    This class implements a convolution operation with configurable input and output channels, kernel size, padding, and stride.
    It leverages Numba CUDA for efficient GPU acceleration.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel_size (int): The size of the convolution kernel.
        padding (Optional[int], optional): The amount of padding to apply. Defaults to 0.
        stride (Optional[int], optional): The stride of the convolution operation. Defaults to 1.
        weight (Optional[torch.Tensor], optional): The initial weight tensor. Defaults to None.
        bias (Optional[torch.Tensor], optional): The initial bias tensor. Defaults to None.

    Example:
        >>> conv = NumbaConv2D(in_channels=3, out_channels=64, kernel_size=3, padding=1, stride=2)
        >>> input_tensor = torch.randn(16, 3, 512, 512, device='cuda')
        >>> output_tensor = conv(input_tensor)
    """
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, weight=None, bias=None):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        if weight is None:
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size, device='cuda'))
        else:
            self.weight = nn.Parameter(weight)

        if bias is None:
            self.bias = nn.Parameter(torch.zeros(out_channels, device='cuda'))
        else:
            self.bias = nn.Parameter(bias)

    def forward(self, x):
        return Conv2dFunction.apply(x, self.weight, self.bias, self.padding, self.stride)


if __name__ == '__main__':
    # Testing by compare the output with torch conv2d module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_tensor = torch.randn(16, 3, 512, 512, device=device, requires_grad=True)

    torch_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=2, stride=2,).to(device)
    torch_output = torch_conv(input_tensor)

    numba_conv = NumbaConv2d(in_channels=3, out_channels=64, kernel_size=3, padding=2, stride=2,
                             weight=torch_conv.weight, bias=torch_conv.bias).to(device)
    numba_output = numba_conv(input_tensor)

    assert ((torch_output - numba_output) < 1e-5).all().item(), "Numba Conv2D output does not match PyTorch Conv2D output"
