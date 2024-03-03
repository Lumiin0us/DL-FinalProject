import torch
from PIL import Image
from torch.utils.data import Dataset
import time 
# Check if MPS is available
device = torch.device('mps')

class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        label, img_path = self.data[idx]

        # Load the image
        img = Image.open(img_path)

        # Apply transformations if provided
        if self.transform:
            img = self.transform(img)

        # Move data to MPS device
        img = img.to(device)
        label = torch.tensor(label, dtype=torch.long).to(device)

        return img, label
    