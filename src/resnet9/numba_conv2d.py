import math
from numba import cuda
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.autograd import Function


@cuda.jit
def conv2d_kernel(input, kernel, output, padding: int, stride: int):
    """
    Performs a 2D convolution operation on a 4D tensor.

    Args:
        input: The input tensor.
        kernel: The convolution kernel.
        output: The output tensor.
        padding (int): The amount of padding to apply.
        stride (int): The stride of the convolution operation.
    """
    combined_idx, out_y, out_x = cuda.grid(3)
    batch_size, in_channels, in_height, in_width = input.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    out_height, out_width = output.shape[2:]

    batch_idx = combined_idx // out_channels
    out_channel_idx = combined_idx % out_channels

    if batch_idx < batch_size and out_channel_idx < out_channels and out_y < out_height and out_x < out_width:
        res = 0.0
        for in_channel in range(in_channels):
            for ky in range(kernel_height):
                for kx in range(kernel_width):
                    in_y = out_y * stride - padding + ky
                    in_x = out_x * stride - padding + kx
                    if 0 <= in_y < in_height and 0 <= in_x < in_width:
                        res += input[batch_idx, in_channel, in_y, in_x] * kernel[out_channel_idx, in_channel, ky, kx]
        output[batch_idx, out_channel_idx, out_y, out_x] = res


class Conv2dFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, weight: Tensor, bias: Optional[Tensor], stride: int, padding: int) -> Tensor:
        ctx.save_for_backward(input, weight, bias)
        ctx.stride = stride
        ctx.padding = padding

        batch_size, in_channels, in_height, in_width = input.shape
        out_channels, _, kernel_height, kernel_width = weight.shape
        out_height = (in_height + 2 * padding - kernel_height) // stride + 1
        out_width = (in_width + 2 * padding - kernel_width) // stride + 1

        output = torch.zeros(batch_size, out_channels, out_height, out_width, device=input.device)

        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            (batch_size * out_channels + threads_per_block[0] - 1) // threads_per_block[0],
            (out_height + threads_per_block[1] - 1) // threads_per_block[1],
            (out_width + threads_per_block[2] - 1) // threads_per_block[2]
        )

        conv2d_kernel[blocks_per_grid, threads_per_block](
            input.detach(), weight.detach(), output, padding, stride
        )

        if bias is not None:
            output += bias.view(1, -1, 1, 1)

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], Optional[Tensor], Optional[Tensor], None, None]:
        input, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = torch.nn.grad.conv2d_input(input.shape, weight, grad_output, stride, padding)

        if ctx.needs_input_grad[1]:
            grad_weight = torch.nn.grad.conv2d_weight(input, weight.shape, grad_output, stride, padding)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum((0, 2, 3))

        return grad_input, grad_weight, grad_bias, None, None


class NumbaConv2d(nn.Module):
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
    
    def __init__(self, 
                 in_channels: int, 
                 out_channels: int,
                 kernel_size: int,
                 padding=0,
                 stride=1,
                 bias=True):
        
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.weight = nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: Tensor):
        return Conv2dFunction.apply(x, self.weight, self.bias, self.stride, self.padding)


if __name__ == '__main__':
    # Testing by compare the output with torch conv2d module
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    input_tensor = torch.randn(16, 3, 512, 512).cuda()

    torch_conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=2, stride=2,).cuda()
    torch_output = torch_conv(input_tensor)

    numba_conv = NumbaConv2d(in_channels=3, out_channels=64, kernel_size=3, padding=2, stride=2).cuda()
    numba_conv.weight.data.copy_(torch_conv.weight)
    numba_conv.bias.data.copy_(torch_conv.bias)
    numba_output = numba_conv(input_tensor)

    assert ((torch_output - numba_output) < 1e-5).all().item(), "Numba Conv2D output does not match PyTorch Conv2D output"
