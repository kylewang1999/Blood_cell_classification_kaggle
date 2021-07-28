cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT

# Testing Blood_cell searched model
python test.py --model_path ./eval-Blood_cell-50Epoch-WAUX-LR0.005/weights.pt --arch DARTS_TS_BC_25EPOCH

# Testing CIFAR searched model
# python test.py --model_path ./eval-CIFAR-50Epoch-WAUX/weights.pt --arch DARTS_CIFAR10_TS_1ST
