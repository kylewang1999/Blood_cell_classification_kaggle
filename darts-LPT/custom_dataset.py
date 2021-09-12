# Dataset setup for colab
import os 
import pandas as pd
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from PIL import Image
import numpy as np
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

class Custom_Dataset(Dataset):
    def __init__(self, images, targets, transform):
        self.features = images
        self.targets = targets
        self.transform = transform
               
    def __len__(self):
        return len(self.targets) 
    
    def __getitem__(self, idx): # Get transformed images
        image = self.features[idx]
        image = Image.open(image)  # Already resized to 128*128
        image = self.transform(image)
        return {
            # 'image': torch.tensor(image, dtype=torch.float),
            'image': image,
            'label': torch.tensor(self.targets[idx], dtype=torch.long)
        }


# Returns train/test/valid dataset as dataframes
def parse_dataset(dataset_path):
    def get_category_id(path):
        category = os.path.split(os.path.split(path)[0])[1]
        CATEGORIES = ['EOSINOPHIL','LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        return CATEGORIES.index(category)
        
    if os.path.isdir(dataset_path + "Train"):
        # Train/Test/Val Paths
        # dataset_path = "../kaggle/blood_cell/"
        Train_Data_Path = Path(dataset_path + "Train").resolve()
        Test_Data_Path = Path(dataset_path + "Test").resolve()
        Validation_Data_Path = Path(dataset_path + "Valid").resolve()
        print(Train_Data_Path)

        # JPG Paths and Lables (Approx. 3000 imgs per type)
        Train_JPG_Path = list(Train_Data_Path.glob(r"**/*.jpeg"))
        Test_JPG_Path = list(Test_Data_Path.glob(r"**/*.jpeg"))
        Validation_JPG_Path = list(Validation_Data_Path.glob(r"**/*.jpeg"))

        Train_JPG_Labels = list(map(get_category_id,Train_JPG_Path))   
        Test_JPG_Labels = list(map(get_category_id,Test_JPG_Path))
        Validation_JPG_Labels = list(map(get_category_id,Validation_JPG_Path))
        print("Total Number of Entries: {}".format(len(Train_JPG_Labels)+len(Test_JPG_Labels)+len(Validation_JPG_Labels)))
        print("Class Counts - Train | Test | Validation")
        print("EOSINOPHIL:   {} | {} | {}".format(Train_JPG_Labels.count(0), Test_JPG_Labels.count(0), Validation_JPG_Labels.count(0)))
        print("LYMPHOCYTE:   {} | {} | {}".format(Train_JPG_Labels.count(1), Test_JPG_Labels.count(1), Validation_JPG_Labels.count(1)))
        print("MONOCYTE  :   {} | {} | {}".format(Train_JPG_Labels.count(2), Test_JPG_Labels.count(2), Validation_JPG_Labels.count(2)))
        print("NEUTROPHIL:   {} | {} | {}".format(Train_JPG_Labels.count(3), Test_JPG_Labels.count(3), Validation_JPG_Labels.count(3)))

        # Convert Paths from List to Seires
        Train_JPG_Path_Series = pd.Series(Train_JPG_Path,name="JPG").astype(str)
        Train_JPG_Labels_Series = pd.Series(Train_JPG_Labels,name="CATEGORY")

        Test_JPG_Path_Series = pd.Series(Test_JPG_Path,name="JPG").astype(str)
        Test_JPG_Labels_Series = pd.Series(Test_JPG_Labels,name="CATEGORY")

        Validation_JPG_Path_Series = pd.Series(Validation_JPG_Path,name="JPG").astype(str)
        Validation_JPG_Labels_Series = pd.Series(Validation_JPG_Labels,name="CATEGORY")

        # Retrieve data from paths, store in dataframes
        Main_Train_Data = pd.concat([Train_JPG_Path_Series,Train_JPG_Labels_Series],axis=1)
        Main_Test_Data = pd.concat([Test_JPG_Path_Series,Test_JPG_Labels_Series],axis=1)
        Main_Validation_Data = pd.concat([Validation_JPG_Path_Series,Validation_JPG_Labels_Series],axis=1)
        
        # Shuffling
        Main_Train_Data = Main_Train_Data.sample(frac=1).reset_index(drop=True)
        Main_Test_Data = Main_Test_Data.sample(frac=1).reset_index(drop=True)
        Main_Validation_Data = Main_Validation_Data.sample(frac=1).reset_index(drop=True)
        print(Main_Train_Data.iloc[0,0])
        print(Main_Train_Data.head(-1))

        return Main_Train_Data, Main_Test_Data, Main_Validation_Data
    
    else:
        Data_Path = Path(dataset_path).resolve()
        JPG_Path = list(Data_Path.glob(r"**/*.jpg"))
        JPG_Labels = list(map(get_category_id, JPG_Path))
        print("Total Number of Entries: {}".format(len(JPG_Labels)))
        print("Class Counts")
        print("EOSINOPHIL: {}".format(JPG_Labels.count(0)))
        print("LYMPHOCYTE: {}".format(JPG_Labels.count(1)))
        print("MONOCYTE  : {}".format(JPG_Labels.count(2)))
        print("NEUTROPHIL: {}".format(JPG_Labels.count(3)))

        JPG_Path_Series = pd.Series(JPG_Path,name="JPG").astype(str)
        JPG_Labels_Series = pd.Series(JPG_Labels,name="CATEGORY")
        Test_Data = pd.concat([JPG_Path_Series, JPG_Labels_Series],axis=1)

        Test_Data = Test_Data.sample(frac=1).reset_index(drop=True)
        print(Test_Data.iloc[0,0])
        print(Test_Data.head(-1))

        return None, Test_Data, None

# Aug. transforms to be applied to train/valid images
def create_transforms():
    MEAN = [0.485, 0.456, 0.406]
    STD = [0.229, 0.224, 0.225]
    train_transform = transforms.Compose([
        # transforms.RandomCrop(128, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    # train_transform = torch.nn.Sequential([
    #     transforms.RandomCrop(128, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    #     transforms.Normalize(MEAN, STD),
    # ])
    # valid_transform =torch.nn.Sequential([
    #     transforms.ToTensor(),
    #     transforms.Normalize(MEAN, STD),
    # ])

    return train_transform, valid_transform

# Return preprocessed train/valid dataset as pytorch DataLoader
def preprocess_data(train_df, valid_df, batch_size, train_search=False):
    train_transform, valid_transform = create_transforms()

    if train_df is not None:
        x_train = train_df.JPG 
        y_train = train_df.CATEGORY
        train_dataset = Custom_Dataset(x_train.values, y_train.values, train_transform) 
    else:
        x_train = None
        y_train = None
        train_dataset = None

    if train_search:    # Portioning Dataset for train_search_ts.py
        train_portion = 0.5
        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(train_portion * num_train))
        train_queue = DataLoader(
            dataset = train_dataset,batch_size = batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[:split]),
            pin_memory=False, num_workers = 4)
        valid_queue = DataLoader(
            dataset = train_dataset, batch_size = batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]),
            pin_memory=False, num_workers = 4)
        external_queue = DataLoader(
            dataset = train_dataset, batch_size = batch_size,
            sampler=torch.utils.data.sampler.SubsetRandomSampler(
                indices[split:num_train]),
            pin_memory=False, num_workers=4)

        print("train_q: {} | valid_q: {} | external_q: {}".format(
            len(train_queue), len(valid_queue), len(external_queue)))
        
        return train_queue, valid_queue, external_queue
        
    else:
        x_valid = valid_df.JPG
        y_valid = valid_df.CATEGORY
        valid_dataset = Custom_Dataset(x_valid.values, y_valid.values, valid_transform)
        
        if train_dataset:
            train_queue = DataLoader(
                dataset = train_dataset, batch_size = batch_size,
                shuffle = True,
                num_workers = 4) 
            print("x_train: {} | x_test/valid: {} ".format(
                len(x_train), len(x_valid)))
        else: 
            train_queue = None
        
        valid_queue = DataLoader(
            dataset = valid_dataset, batch_size = batch_size,
            shuffle = False,
            num_workers = 4)

        return train_queue, valid_queue


