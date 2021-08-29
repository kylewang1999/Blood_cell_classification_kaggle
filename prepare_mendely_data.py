import os
import numpy as np
from typing import List
import shutil
import torch
from PIL import Image
from torchvision import transforms


data_dir = '../snkd93bnjr-1/PBC_dataset_normal_DIB/PBC_dataset_normal_DIB'
# Make sure train split if first in the list
splits = [(0.64,'Train'),(0.24,'Val'),(0.12,'Test')]
final_path = '../mendely_data3'



def cp_files(files : List[str],src_dir : str, des_dir : str) -> None:
    for file in files:
        shutil.copy(os.path.join(src_dir, file), os.path.join(des_dir,file),)



if not os.path.exists(final_path):
    os.makedirs(final_path, exist_ok=True)
for split in splits:
    os.makedirs(os.path.join(final_path, split[1]),exist_ok = True)

##### First Create the data splits #######

labels = os.listdir(data_dir)
labels.sort()

np.random.seed(42)
torch.manual_seed(42)


for label in labels:
    images = os.listdir(os.path.join(data_dir, label))
    images.sort()
    np.random.shuffle(images)
    
    prev_index = 0

    for split in splits:
    
        final_label_path = os.path.join(final_path, split[1], label)
        if not os.path.exists(final_label_path):
            os.makedirs(final_label_path,exist_ok = True)

        end_index = int(split[0] * len(images)) + prev_index
        print(end_index - prev_index)
        cp_files(images[prev_index:end_index], src_dir = os.path.join(data_dir, label), des_dir = final_label_path)
        prev_index = end_index

print('Data Transfer Complete')
##########################################


##### Next Augment the Dataset #######

data_dir = os.path.join(final_path, splits[0][1])
images_per_label = 4096


augment_transform = transforms.Compose([transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip() , transforms.RandomRotation(60)])

def augment_and_save(image : str, src_dir : str, i = 0) -> None:
    img = Image.open(os.path.join(src_dir, image))
    new_img : Image = augment_transform(img)
    new_img.save(os.path.join(src_dir, os.path.splitext(image)[0] + "_" + str(i) + os.path.splitext(image)[1]))
    pass

labels = os.listdir(data_dir)
labels.sort()
for label in labels:
    image_folder = os.path.join(data_dir, label)
    images = os.listdir(image_folder)
    images.sort()
    tot = len(images)
    i = 0 
    print('Augmenting Label: ' + label)

    while tot+len(images) <= images_per_label:
        for image in images:
            augment_and_save(image,image_folder, i = i)
            tot+=1
        print(f'Done Images: {tot}/{images_per_label}')
        i+=1
    diff = images_per_label - tot
    np.random.shuffle(images)
    for image in images[:diff]:
        augment_and_save(image,image_folder, i = i)
print('Completed Augmenting of Labels')

############################################