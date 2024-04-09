from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch

class GenericDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.images = []
        self.image_paths = sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(".png")])
    
    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")
        if self.transform:
            image = self.transform(image)
        return image, os.path.basename(image_path)  # Return the image and its filename

def CreateDatasetSynthesis(phase, input_path, contrast1='POR', contrast2='DEL'):
    # Define the transformations
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convert PIL Image to PyTorch tensor
        transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
    ])

    por_dataset = GenericDataset(os.path.join(input_path, contrast1), transform=transform)
    del_dataset = GenericDataset(os.path.join(input_path, contrast2), transform=transform)

    # Ensure that both datasets have the same number of images
    assert len(por_dataset) == len(del_dataset), "The number of images in each dataset must be the same"

    combined_dataset = torch.utils.data.TensorDataset(
        torch.stack([por_dataset[i][0] for i in range(len(por_dataset))]),
        torch.stack([del_dataset[i][0] for i in range(len(del_dataset))])
    )
    paths = list(zip([por_dataset[i][1] for i in range(len(por_dataset))], 
                     [del_dataset[i][1] for i in range(len(del_dataset))]))

    return combined_dataset, paths
