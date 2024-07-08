import math
import numpy as np
from numba import cuda

import torch
from torch import nn


@cuda.jit
def conv2d_kernel(input, kernel, output):
    batch_idx, out_y, out_x = cuda.grid(3)
    if batch_idx < input.shape[0] and out_y < output.shape[2] and out_x < output.shape[3]:
        for out_channel in range(output.shape[1]):
            sum = 0.0
            for in_channel in range(input.shape[1]):
                for ky in range(kernel.shape[2]):
                    for kx in range(kernel.shape[3]):
                        in_y = out_y + ky
                        in_x = out_x + kx
                        if in_y < input.shape[2] and in_x < input.shape[3]:
                            sum += (input[batch_idx, in_channel, in_y, in_x] *
                                    kernel[out_channel, in_channel, ky, kx])
            output[batch_idx, out_channel, out_y, out_x] = sum

class NumbaConv2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super(NumbaConv2D, self).__init__()
        self.kernel = torch.nn.Parameter(torch.randn(out_channels, in_channels, kernel_size, kernel_size))
        self.bias = torch.nn.Parameter(torch.zeros(out_channels))

    def forward(self, x):
        assert x.is_cuda, "Input must be a CUDA tensor"
        assert x.dim() == 4, "Input must be a 4D tensor"

        # Detach input and kernel for CUDA kernel
        x_detached = x.detach()
        kernel_detached = self.kernel.detach()

        batch_size, in_channels, in_height, in_width = x.shape
        out_channels, _, kernel_size, _ = self.kernel.shape
        out_height = in_height - kernel_size + 1
        out_width = in_width - kernel_size + 1

        output = torch.zeros(batch_size, out_channels, out_height, out_width, device=x.device)

        threads_per_block = (8, 8, 8)
        blocks_per_grid = (
            math.ceil(batch_size / threads_per_block[0]),
            math.ceil(out_height / threads_per_block[1]),
            math.ceil(out_width / threads_per_block[2])
        )

        conv2d_kernel[blocks_per_grid, threads_per_block](
            x_detached, kernel_detached, output
        )

        # Instead of modifying output in-place, create a new tensor
        return output + self.bias.view(1, -1, 1, 1)
    

if __name__ == '__main__':
    # Usage example
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conv_layer = NumbaConv2D(in_channels=3, out_channels=64, kernel_size=3).to(device)
    input_tensor = torch.randn(16, 3, 512, 512, device=device, requires_grad=True)
    output = conv_layer(input_tensor)
    print(output.shape)

    # Test backward pass
    loss = output.sum()
    loss.backward()
    print("Gradient computed:", input_tensor.grad is not None)