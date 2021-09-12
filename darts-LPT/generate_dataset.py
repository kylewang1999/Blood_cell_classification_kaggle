# Another BCCD Dataset that contain correct labeling for test dataset
import os 
import pandas as pd
import numpy as np
from pathlib import Path
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import cv2
from PIL import Image

from scipy.spatial import distance as dist
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm 
# from sklearn.utils import shuffle
from sklearn import decomposition
from sklearn.model_selection import train_test_split #
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
import keras
from keras.applications.vgg16 import VGG16 
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Sequential, Model 
from keras.applications import DenseNet201
from keras.initializers import he_normal
from keras.layers import Lambda, SeparableConv2D, BatchNormalization, Dropout, MaxPooling2D, Input, Dense, Conv2D, Activation, Flatten 
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import imutils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def main():
    dataset_path='./wbc_images_resized/'
    Train_Data_Path = Path(dataset_path + "TRAIN")
    Test_Data_Path = Path(dataset_path + "TEST")
    Validation_Data_Path = Path(dataset_path + "TEST_SIMPLE")
    # If working on local, get absolute paths
    Train_Data_Path = Train_Data_Path.resolve()
    Test_Data_Path = Test_Data_Path.resolve()
    Validation_Data_Path = Validation_Data_Path.resolve()
    print(Train_Data_Path)

    # JPG Paths and Lables (Approx. 3000 imgs per type)
    Train_JPG_Path = list(Train_Data_Path.glob(r"**/*.jpeg"))
    Test_JPG_Path = list(Test_Data_Path.glob(r"**/*.jpeg"))
    Validation_JPG_Path = list(Validation_Data_Path.glob(r"**/*.jpeg"))

    def get_category_id(path):
        category = os.path.split(os.path.split(path)[0])[1]
        CATEGORIES = ['EOSINOPHIL','LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
        return CATEGORIES.index(category)

    Train_JPG_Labels = list(map(get_category_id,Train_JPG_Path))   
    Test_JPG_Labels = list(map(get_category_id,Test_JPG_Path))
    Validation_JPG_Labels = list(map(get_category_id,Validation_JPG_Path))
    
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
    Main_Data = pd.concat(Main_Train_Data, Main_Test_Data, Main_Validation_Data, axis=0)

    # Shuffling
    Main_Data = Main_Data.sample(frac=1).reset_index(drop=True)
    print("Total Number of Entries: {}".format(len(Main_Data)))
    print(Main_Data.iloc[0,0])
    print(Main_Data.head(-1))

    split_train_test = int(np.floor(0.75 * len(Main_Data)))
    Train_Data = Main_Data.iloc[:split_train_test,:]
    Test_Valid_Data = Main_Data.iloc[split_train_test:,:]

    split_test_valid = int(np.floor(0.80 * len(Test_Valid_Data)))
    Test_Data = Test_Valid_Data.iloc[:split_test_valid,:]
    Valid_Data = Test_Valid_Data.iloc[split_test_valid:,:]

    DSET_TYPES = ['Train/', 'Test/', 'Valid/']
    DSET = [Train_Data, Test_Data, Valid_Data]
    CATEGORIES = ['EOSINOPHIL','LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']
    for dset_type, dset in zip(DSET_TYPES, DSET):
        os.mkdir('./wbc_reorg/' + dset_type)
        for c in CATEGORIES:
            os.mkdir('./wbc_reorg/' + dset_type + c)
        for row in dset:
            path = row['JPG']
            label = row['CATEGORY']
            image = Image.open(path)
            if label == 0:
                image.save('./wbc_reorg/' + dset_type + 'EOSINOPHIL')
            elif label == 1:
                image.save('./wbc_reorg/' + dset_type +'LYMPHOCYTE')
            elif label == 2:
                image.save('./wbc_reorg/' + dset_type +'MONOCYTE')
            else:
                image.save('./wbc_reorg/' + dset_type +'NEUTROPHIL')

    print("DONE!")


