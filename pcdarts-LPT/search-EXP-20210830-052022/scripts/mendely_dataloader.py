# Sample Usage:
# from mendely_dataloader import get_dataloaders
# train_l , val_l , test_l = get_dataloaders('path_to_prepared_data', batch_size = 16, num_workers = 2)


import os
import torch.utils.data as data
import torch
import torchvision
from torchvision import transforms
from typing import List


def get_dataloaders(data_dir : str = '../mendely_data', batch_size : int = 4, num_workers : int = 4, train_search = False) -> List[data.DataLoader]:


    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
        ])

    splits = ['Train','Val', 'Test']

    dataloaders = []

    if not train_search:
        for split in splits:
            dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, split), transform=transform)
            dataloaders.append(data.DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers))
    else:
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'Train'), transform=transform)
        split = int(0.5  * len(dataset))
        indices = list(range(len(dataset)))
        dataloaders.append(data.DataLoader(dataset, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), batch_size=batch_size, num_workers=num_workers))
        dataloaders.append(data.DataLoader(dataset, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]), batch_size=batch_size, num_workers=num_workers))
        dataloaders.append(data.DataLoader(dataset, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]), batch_size=batch_size, num_workers=num_workers))
    return dataloaders
