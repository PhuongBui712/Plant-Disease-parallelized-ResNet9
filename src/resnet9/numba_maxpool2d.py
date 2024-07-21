import math
from numba import cuda
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch import nn
from torch.autograd import Function


MIN_FLOAT32 = torch.finfo(torch.float32).min

@cuda.jit
def max_pool_2d_kernel(input, output, kernel_size: int, padding: int, stride: int):
    """
    Performs a 2D max pooling operation on a 4D tensor.

    Args:
        input: The input tensor.
        output: The output tensor.
        kernel_size (int): The size of the pooling kernel.
        padding (int): The amount of padding to apply.
        stride (int): The stride of the pooling operation.
    """
    idx, out_h, out_w = cuda.grid(3)
    
    batch_idx = idx // input.shape[1]
    channel = idx % input.shape[1]
    
    if batch_idx < input.shape[0] and channel < input.shape[1] and out_h < output.shape[2] and out_w < output.shape[3]:
        max_val = MIN_FLOAT32
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                in_y = out_h * stride - padding + ky
                in_x = out_w * stride - padding + kx
                if 0 <= in_y < input.shape[2] and 0 <= in_x < input.shape[3]:
                    max_val = max(max_val, input[batch_idx, channel, in_y, in_x])
        output[batch_idx, channel, out_h, out_w] = max_val


@cuda.jit
def max_pool_2d_backward_kernel(input, output, grad_output, grad_input, kernel_size: int, padding: int, stride: int):
    """
    Performs the backward pass for a 2D max pooling operation on a 4D tensor.

    This kernel calculates the gradient of the input tensor based on the gradient of the output tensor and the
    pooling operation's parameters. It uses atomic addition to accumulate gradients for elements that contributed
    to the maximum value in the pooling window.

    Args:
        input: The input tensor.
        output: The output tensor.
        grad_output: The gradient of the output tensor.
        grad_input: The gradient of the input tensor (to be accumulated).
        kernel_size (int): The size of the pooling kernel.
        padding (int): The amount of padding applied during the forward pass.
        stride (int): The stride of the pooling operation.
    """
    idx, in_h, in_w = cuda.grid(3)
    
    batch_idx = idx // input.shape[1]
    channel = idx % input.shape[1]
    
    if batch_idx < input.shape[0] and channel < input.shape[1] and in_h < input.shape[2] and in_w < input.shape[3]:
        for ky in range(kernel_size):
            for kx in range(kernel_size):
                out_h = (in_h + padding - ky) // stride
                out_w = (in_w + padding - kx) // stride
                if 0 <= out_h < output.shape[2] and 0 <= out_w < output.shape[3]:
                    if input[batch_idx, channel, in_h, in_w] == output[batch_idx, channel, out_h, out_w]:
                        cuda.atomic.add(grad_input, (batch_idx, channel, in_h, in_w), grad_output[batch_idx, channel, out_h, out_w])


class MaxPool2dFunction(Function):
    @staticmethod
    def forward(ctx, input: Tensor, kernel_size: int, stride: int, padding: int) -> Tensor:
        ctx.save_for_backward(input)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        ctx.padding = padding

        # Detach input for CUDA operations
        input_data = input.detach()

        batch_size, channels, in_height, in_width = input.shape
        out_height = (in_height + 2 * padding - kernel_size) // stride + 1
        out_width = (in_width + 2 * padding - kernel_size) // stride + 1

        output = torch.full((batch_size, channels, out_height, out_width), MIN_FLOAT32, device=input.device)

        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            math.ceil(batch_size * channels / threads_per_block[0]),
            math.ceil(out_height / threads_per_block[1]),
            math.ceil(out_width / threads_per_block[2])
        )

        max_pool_2d_kernel[blocks_per_grid, threads_per_block](
            input_data, output, kernel_size, padding, stride
        )

        return output

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tuple[Optional[Tensor], None, None, None]:
        input, = ctx.saved_tensors
        kernel_size = ctx.kernel_size
        stride = ctx.stride
        padding = ctx.padding

        # Detach tensors for CUDA operations
        input_data = input.detach()
        grad_output_data = grad_output.detach()

        grad_input = torch.zeros_like(input)

        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            math.ceil(input.shape[0] * input.shape[1] / threads_per_block[0]),
            math.ceil(input.shape[2] / threads_per_block[1]),
            math.ceil(input.shape[3] / threads_per_block[2])
        )

        output = MaxPool2dFunction.forward(ctx, input, kernel_size, stride, padding)

        max_pool_2d_backward_kernel[blocks_per_grid, threads_per_block](
            input_data, output, grad_output_data, grad_input, kernel_size, padding, stride
        )

        return grad_input, None, None, None


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
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x: Tensor):
        return MaxPool2dFunction.apply(x, self.kernel_size, self.stride, self.padding)


if __name__ == '__main__':
    input = torch.randn(1, 3, 256, 256, device='cuda')

    torch_max_pooling = nn.MaxPool2d(kernel_size=3, stride=2).cuda()
    torch_output = torch_max_pooling(input)

    numba_max_pooling = NumbaMaxPool2d(kernel_size=3, stride=2).cuda()
    numba_output = numba_max_pooling(input)

    assert ((torch_output - numba_output) < 1e-6).all().item(), "Numba MaxPool2D output does not match PyTorch MaxPool2D output"
