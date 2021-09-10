# Sample Usage:
# from mendely_dataloader import get_dataloaders
# train_l , val_l , test_l = get_dataloaders('path_to_prepared_data', batch_size = 16, num_workers = 2)

import os
import torch.utils.data as data
import torch
import torchvision
from torchvision import transforms
import numpy as np
from typing import List
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset


def get_dataloaders(data_dir : str = '../kaggle/PBC_dataset_split/PBC_dataset_split', batch_size : int = 4, num_workers : int = 2, train_search = False) -> List[data.DataLoader]:

    transform = transforms.Compose([
        transforms.Resize((128,128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225] )
        ])

    

    # 64% | 24% | 12%
    # splits = ['Train','Val', 'Test']
    splits = ['Train','Test', 'Val']
    dataloaders = []

    if data_dir == '../kaggle/BCCD_Dataset/BCCD_410/BCCD_410_split':
        # Using BCCD_410 Dataset only for testing
        dataloaders = [None, None]
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir), transform=transform)
        dataloaders.append(data.DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers))
        return dataloaders

    if not train_search:
        for split in splits:
            # Full Train/Test/Valid set containing all 8 types of PBC
            dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, split), transform=transform)
            
            # wbc_types = ['0_eosinophil', '1_lymphocyte', '2_monocyte', '3_neutrophil']
            # wbc_types_idx = [dataset.class_to_idx[t] for t in wbc_types] 
            # print(wbc_types_idx)
            # # Select only 4 types of WBC & Put them in appropriate subset
            # idx = [i for i in range(len(dataset)) if dataset.imgs[i][1] in wbc_types_idx]
            # subset = Subset(dataset, idx)
            # dataloaders.append(data.DataLoader(subset, batch_size=batch_size, shuffle=True,  num_workers=num_workers))
            dataloaders.append(data.DataLoader(dataset, batch_size=batch_size, shuffle=True,  num_workers=num_workers))
    else:
        dataset = torchvision.datasets.ImageFolder(root=os.path.join(data_dir, 'Train'), transform=transform)
        split = int(0.5  * len(dataset))
        indices = list(range(len(dataset)))
        np.random.shuffle(indices)
        # Train queue
        dataloaders.append(data.DataLoader(dataset, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]), batch_size=batch_size, num_workers=num_workers))
        # Valid queue
        dataloaders.append(data.DataLoader(dataset, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]), batch_size=batch_size, num_workers=num_workers))
        # External queue
        dataloaders.append(data.DataLoader(dataset, sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:]), batch_size=batch_size, num_workers=num_workers))
    return dataloaders
