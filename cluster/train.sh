#!/usr/bin/env bash
nvidia-smi
conda env create -f environment.yml
# source activate yxy
# pip install Pillow==6.2.2
# pip install pydicom
# pip install --upgrade efficientnet-pytorch
python test.py --auxiliary --model_path cifar10_model.pt
