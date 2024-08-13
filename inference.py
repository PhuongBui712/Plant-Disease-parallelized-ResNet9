import os
import json
from PIL import Image
from typing import Optional
import torch
from torch.nn import Module
from torchvision.transforms import ToTensor

from src.resnet9.pytorch_resnet9 import ConvBlock, ResNet9


def image_to_tensor(img_path: str, batched: bool = True) -> torch.Tensor:
    """Loads an image from a file path and converts it to a PyTorch tensor.

    Args:
        img_path (str): The path to the image file.
        batched (bool, optional): Whether to add a batch dimension to the tensor. Defaults to True.

    Returns:
        torch.Tensor: The image tensor.
    """
    img = Image.open(img_path)
    transformer = ToTensor()
    img_tensor = transformer(img)
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


if __name__ == '__main__':
    # load model
    model = torch.load(os.path.join(os.path.dirname(__file__), './model/numba_resnet9.pt'),
                       map_location=torch.device('cpu'))

    # load label name map
    with open('./data/class_idx.json', 'r') as f:
        label_name = json.load(f)
    label_name = {v: k for k, v in label_name.items()}

    # inference
    img_path = os.path.join(os.path.dirname(__file__),
                            'data/New-Plant-Diseases-Dataset/test/test/AppleCedarRust1.JPG')
    
    pred = inference(model, img_path, label_name)
    print(pred)