from typing import Optional, Callable, Tuple
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder
from torchvision import transforms


class PlantDiseaseDataset(Dataset):
    """
    A PyTorch Dataset class for plant disease classification.

    This class loads images from a specified directory and applies optional transformations.
    It assumes the directory structure follows the ImageFolder convention, where each subdirectory
    represents a different disease class.

    If no transformations are provided (`transforms` is None), the class will convert the images
    to PyTorch tensors by default.

    Args:
        path (str): The path to the directory containing the plant disease images.
        transforms (Callable, optional): A callable object (e.g., torchvision.transforms)
            to apply to the images. Defaults to None.
    """
    def __init__(self,
                 path: str,
                 transform_function: Optional[Callable] = None) -> None:
        super().__init__()

        transform = transform_function or transforms.ToTensor()
        self.img_folder = ImageFolder(path, transform=transform)

    def __len__(self) -> int:
        return len(self.img_folder)
    
    def __getitem__(self, idx) -> Tuple[Tensor, int]:
        return self.img_folder[idx]