#!/usr/bin/env bash
nvidia-smi
# conda env create -f environment.yml
# source activate yxy
# pip install Pillow==6.2.2
# pip install pydicom
# pip install --upgrade efficientnet-pytorch
pwd
python ../darts-LPT/train_custom_colab.py --auxiliary --model_path --epochs 50 --save eval-EXP-CIFAR-50 
