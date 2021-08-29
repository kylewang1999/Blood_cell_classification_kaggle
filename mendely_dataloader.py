# Sample Usage:
# from mendely_dataloader import get_dataloaders
# train_l , val_l , test_l = get_dataloaders('path_to_prepared_data', batch_size = 16, num_workers = 2)


import os
import torch.utils.data as data
import torchvision
from torchvision import transforms
from typing import List

def get_dataloaders(data_dir : str = '../mendely_data', batch_size : int = 4, num_workers : int = 4) -> List[data.DataLoader]:


    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
        ])

    splits = ['Train','Val', 'Test']

    dataloaders = []

    for split in splits:
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, split), transform=transform)
        dataloaders.append(data.DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers))
    
    return dataloaders
