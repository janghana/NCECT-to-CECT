from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import os
import numpy as np
import torch
from sklearn.model_selection import train_test_split

### ### Arterial Phase Dataset ### ###
class ArterialPhaseDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.images = []

        # List all PNG files in the directory and sort them
        image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])

        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            image = Image.open(image_path).convert("L") # Convert to Gray Scale
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

### ### Portal Venous Phase Dataset ### ###
class PortalVenousPhaseDataset(Dataset):
    def __init__(self, directory, transform=None):
        self.transform = transform
        self.images = []

        # List all PNG files in the directory and sort them
        image_files = sorted([f for f in os.listdir(directory) if f.endswith('.png')])

        for image_file in image_files:
            image_path = os.path.join(directory, image_file)
            image = Image.open(image_path).convert("L") # Convert to Gray Scale
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image

def CreateDatasetSynthesis(phase, input_path, contrast1 = 'ART', contrast2 = 'POR'):
    ### ### Define the transformations ### ###
    assert phase in ['train', 'val', 'test'], "Phase should be 'train' or 'val' or 'test'"

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    ### ### Load Dataset ### ###
    art_target_file = input_path + "/ART/"
    data_fs_art = ArterialPhaseDataset(art_target_file, transform=transform)

    por_target_file = input_path + "/POR/"
    data_fs_por = PortalVenousPhaseDataset(por_target_file, transform=transform)

    ### ### Split the Dataset ### ###
    if phase in ['train', 'val']:
        total_samples = len(data_fs_art)
        num_train = int(total_samples * 0.75)
        num_val = total_samples - num_train

        indices = list(range(total_samples))
        train_indices, val_indices = train_test_split(indices, train_size=num_train, test_size=num_val, shuffle=True, random_state=42)

        if phase == 'train':
            selected_indices = train_indices
        elif phase == 'val':
            selected_indices = val_indices

        data_fs_art = torch.stack([data_fs_art[i] for i in selected_indices])
        data_fs_por = torch.stack([data_fs_por[i] for i in selected_indices])

    elif phase == 'test':
        selected_indices = list(range(len(data_fs_art)))
        data_fs_art = torch.stack([data_fs_art[i] for i in selected_indices])
        data_fs_por = torch.stack([data_fs_por[i] for i in selected_indices])

    dataset = torch.utils.data.TensorDataset(data_fs_art, data_fs_por)
    return dataset
