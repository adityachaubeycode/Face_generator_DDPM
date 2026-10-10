import os
import torch
from PIL import Image
from torch.utils.data import Dataset, dataloader

class CelebA(Dataset):
  def __init__(self, root, transform=None):
    self.root = root
    # Filter for common image extensions to ensure only image files are loaded
    self.image_files = [f for f in os.listdir(root) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
    self.transform = transform

  def __len__(self):
    return len(self.image_files)

  def __getitem__(self, idx):
    img_path = os.path.join(self.root, self.image_files[idx])
    img = Image.open(img_path).convert('RGB')
    if self.transform:
      img = self.transform(img)
    return img

def get_loaders(data_path, batch_size, shuffle=True, data_transform=None, num_workers=4):
    dataset = CelebA(root=data_path, transform=data_transform)
    dataloader = torch.utils.data.DataLoader(dataset,
    batch_size=batch_size,
    shuffle=shuffle,
    num_workers=num_workers)
    print(f"Total lenght of the dataset:{len(dataset)}")
    print(f"Total lenght of the dataloader:{len(dataloader)}")
    return dataset, dataloader