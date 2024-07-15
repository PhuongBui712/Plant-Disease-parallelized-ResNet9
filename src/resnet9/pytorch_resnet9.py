import torch
from torch import Tensor
from torch import nn


class ConvBlock(nn.Module):
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
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        if pooling:
            self.conv.append(nn.MaxPool2d(kernel_size=pooling_kernel))

    def forward(self, X: Tensor):
        return self.conv(X)
    

class ResNet9(nn.Module):
    """
    A ResNet-9 model implemented using PyTorch.

    This class implements a ResNet-9 architecture with configurable input channels and number of classes.
    It uses standard PyTorch modules for convolutional layers, batch normalization, ReLU activation, max pooling, and linear layers.

    Args:
        in_channels (int): The number of input channels.
        num_classes (int): The number of output classes.

    Example:
        >>> model = ResNet9(in_channels=3, num_classes=10)
        >>> input_tensor = torch.randn(16, 3, 224, 224)
        >>> output_tensor = model(input_tensor)

    Input:
        A 4D tensor of shape (batch_size, in_channels, height, width) representing the input images.

    Output:
        A 2D tensor of shape (batch_size, num_classes) representing the predicted class probabilities.
    """
    def __init__(self,
                 in_channels: int,
                 num_classes: int,) -> None:
        super().__init__()

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, pooling=True)

        self.residual1 = nn.Sequential(
            ConvBlock(128, 128),
            ConvBlock(128, 128)
        )

        self.conv3 = ConvBlock(in_channels=128, out_channels=256, pooling=True)
        self.conv4 = ConvBlock(in_channels=256, out_channels=512, pooling=True)

        self.residual2 = nn.Sequential(
            ConvBlock(512, 512),
            ConvBlock(512, 512)
        )

        self.classifier = nn.Sequential(
            nn.MaxPool2d(4),
            nn.Flatten(),
            nn.Linear(in_features=512, out_features=num_classes)
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