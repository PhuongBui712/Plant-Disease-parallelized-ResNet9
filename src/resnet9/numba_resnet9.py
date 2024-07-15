import torch
from torch import Tensor
from torch import nn

from numba_conv2d import NumbaConv2d
from numba_batchnorm2d import NumbaBatchNorm2d
from numba_relu import NumbaReLU
from numba_maxpool2d import NumbaMaxPool2d
from numba_linear import NumbaLinear


class NumbaConvBlock(nn.Module):
    """
    A convolutional block with optional max pooling.

    Args:
        in_channels (int): The number of input channels.
        out_channels (int): The number of output channels.
        kernel (int, optional): The size of the convolution kernel. Defaults to 3.
        stride (int, optional): The stride of the convolution operation. Defaults to 1.
        padding (int, optional): The amount of padding to apply. Defaults to 1.
        pooling (bool, optional): Whether to apply max pooling after the convolution. Defaults to False.
        pooling_kernel (int, optional): The size of the max pooling kernel. Defaults to 4.

    Returns:
        torch.Tensor: The output tensor.
    """
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel: int = 3,
                 stride: int = 1,
                 padding: int = 1,
                 pooling: bool = False,
                 pooling_kernel: int = 4) -> None:
    
        super().__init__()

        self.conv = nn.Sequential(
            NumbaConv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            NumbaBatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if pooling:
            self.conv.append(NumbaMaxPool2d(kernel_size=pooling_kernel))

    def forward(self, X: Tensor):
        return self.conv(X)
    

class NumbaResNet9(nn.Module):
    """
    A ResNet-9 model implemented using Numba CUDA for efficient GPU acceleration.

    This class implements a ResNet-9 architecture with configurable input channels and number of classes.
    It leverages Numba CUDA for efficient GPU acceleration, providing significant performance gains compared to standard PyTorch implementations.

    Args:
        in_channels (int): The number of input channels.
        num_classes (int): The number of output classes.

    Example:
        >>> model = NumbaResNet9(in_channels=3, num_classes=10)
        >>> input_tensor = torch.randn(16, 3, 224, 224, device='cuda')
        >>> output_tensor = model(input_tensor)
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,) -> None:
        super().__init__()

        self.conv1 = NumbaConvBlock(in_channels=in_channels, out_channels=64)
        self.conv2 = NumbaConvBlock(in_channels=64, out_channels=128, pooling=True)
        
        self.residual1 = nn.Sequential(
            NumbaConvBlock(128, 128),
            NumbaConvBlock(128, 128)
        )

        self.conv3 = NumbaConvBlock(in_channels=128, out_channels=256, pooling=True)
        self.conv4 = NumbaConvBlock(in_channels=256, out_channels=512, pooling=True)
        
        self.residual2 = nn.Sequential(
            NumbaConvBlock(512, 512),
            NumbaConvBlock(512, 512)
        )
        
        self.classifier = nn.Sequential(
            NumbaMaxPool2d(4),
            nn.Flatten(),
            NumbaLinear(in_features=512, out_features=num_classes)
        )

    def forward(self, x: Tensor):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.residual1(x) + x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.residual2(x) + x
        x = self.classifier(x)

        return x