from flask import Flask, request, jsonify
from ultralytics import YOLO
import numpy as np
import cv2
from flask_cors import CORS

import os
import re
import json
from PIL import Image
from typing import Optional
from torch.nn import Module
from torchvision.transforms import ToTensor
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


def image_to_tensor(imgage, batched: bool = True) -> torch.Tensor:
    """Loads an image from a file path and converts it to a PyTorch tensor.
    Args:
        img_path (str): The path to the image file.
        batched (bool, optional): Whether to add a batch dimension to the tensor. Defaults to True.
    Returns:
        torch.Tensor: The image tensor.
    """
    #img = Image.open(img_path)
    transformer = ToTensor()
    img_tensor = transformer(imgage)
    if batched:
        img_tensor = img_tensor.unsqueeze(0)

    return img_tensor


def inference(model: Module, img_path: str, label_name_map: Optional[dict] = None, device: str = 'cpu'):
    """Performs inference on an image using a given model.
    Args:
        model (Module): The PyTorch model to use for inference.
        img_path (str): The path to the image file.
        label_name_map (Optional[dict], optional): A dictionary mapping prediction indices to label names. Defaults to None.
        device (str, optional): The device to use for inference (e.g., 'cpu', 'cuda'). Defaults to 'cpu'.
    Returns:
        Union[int, str]: The predicted class index or label name, depending on whether `label_name_map` is provided.
    """
    # load image 
    img_tensor = image_to_tensor(img_path)
    img_tensor = img_tensor.to(device)

    # inference
    model.to(device)
    model.eval()
    with torch.no_grad():
        output_logit = model(img_tensor)

    prediction_idx = output_logit.argmax(dim=1).item()

    # get label name
    if label_name_map is not None:
        return label_name_map[prediction_idx]
    return prediction_idx

# Khởi tạo Flask app
app = Flask(__name__)
CORS(app)

# Load pretrained YOLOv8n model

model = torch.load('./numba_resnet9.pt',
                    map_location=torch.device('cpu'))

with open('./class_idx.json', 'r') as f:
        label_name = json.load(f)
label_name = {v: k for k, v in label_name.items()}

@app.route('/process_traffic_status', methods=['POST'])
def process_traffic_status():
    # Nhận hình ảnh từ request
    img_data = request.files['image'].read()
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    
    # Tạo từ điển để lưu số lượng các vật thể của mỗi loại
    pred = inference(model, img, label_name)

    processed_output = re.sub(r'_+', ' ', pred)

    return jsonify({'traffic_status': processed_output.title()})

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000)