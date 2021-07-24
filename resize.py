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

# dataset_path = '/Users/KyleWang/Desktop/Blood_cell_classification_kaggle/kaggle/blood_cell'
dataset_path = './kaggle/blood_cell/'
Train_Data_Path = Path(dataset_path + "dataset2-master/dataset2-master/images/TRAIN")
Test_Data_Path = Path(dataset_path + "dataset2-master/dataset2-master/images/TEST")
Validation_Data_Path = Path(dataset_path + "dataset2-master/dataset2-master/images/TEST_SIMPLE")

# im = Image.open('./example_image.jpeg').resize((128,128))
# im.save('example3.jpeg')
print(Test_Data_Path)
# JPG Paths and Lables (Approx. 3000 imgs per type)
Train_JPG_Path = list(Train_Data_Path.glob(r"**/*.jpeg"))
for path in Train_JPG_Path:
    im = Image.open(path).resize((128,128))
    im.save(path)    
Test_JPG_Path = list(Test_Data_Path.glob(r"**/*.jpeg"))
for path in Test_JPG_Path:
    im = Image.open(path).resize((128,128))
    im.save(path) 
Validation_JPG_Path = list(Validation_Data_Path.glob(r"**/*.jpeg"))
for path in Validation_JPG_Path:
    im = Image.open(path).resize((128,128))
    im.save(path) 

print("DONE!")