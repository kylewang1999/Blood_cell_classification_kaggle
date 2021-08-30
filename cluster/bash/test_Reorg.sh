cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT

# Testing Blood_cell searched model
python test.py --model_path ./eval-Blood_cell-50Epoch-WAUX-LR0.005/weights.pt --arch DARTS_TS_BC_25EPOCH --batch_size 3

# Testing CIFAR searched model
# python test.py --model_path ./eval-CIFAR-50Epoch-WAUX/weights.pt --arch DARTS_CIFAR10_TS_1ST

# cf100-300
# cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/eval-CF100-300-20210801-012406# pwd/k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/eval-CF100-300-20210801-012406
# python test.py --model_path ./eval-CF100-300-20210801-012406/weights.pt --arch DARTS_CIFAR100_TS_1ST --batch_size 8 --layers 12
    # 08/02 10:31:50 AM test_acc 71.531966

# cf10-300
# cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/
# python test.py --model_path ./eval-CF10-300-20210801-012406/weights.pt --arch DARTS_CIFAR10_TS_1ST --batch_size 8 --layers 12
    # 08/03 05:27:47 AM test_acc 62.042622


    

# python test.py --model_path ./eval-BC-50-300-20210808-070655/weights.pt --arch DARTS_TS_BC_50EPOCH --batch_size 8 --layers 12 

# Official Darts 50 Epoch Search
# python test.py --model_path ./eval-OFF-BC-50-300-20210818-014314/weights.pt --arch DARTS_BC_50EPOCH --batch_size 8 --layers 12 
    # 6 Epoch
    # 08/18 02:43:15 AM test_acc 86.288701
    # 300 Epoch
    # 08/19 01:53:04 PM test_acc 88.379574


# Hybrid Darts-LPT
# python test.py --model_path ./eval-darts-hybrid-reorg-20210823-122343/weights.pt --arch DARTS_TS_BC_50EPOCH --batch_size 8 --layers 12 
python test.py --model_path ./eval-darts-hybrid-reorg-20210828-063008/weights.pt --arch DARTS_TS_BC_50EPOCH --batch_size 8 --layers 12 
    # 22 Epochs
    # 08/23 03:41:37 PM test_acc 96.364363
    # 290 Epoch
    # 08/25 02:00:38 AM test_acc 100.000000


# BC-50-300
# python test.py --model_path ./eval-BC-50-300-Reorg-20210823-122141/weights.pt --arch DARTS_TS_BC_50EPOCH --batch_size 8 --layers 12
    # ~150 Epoch
    # 08/24 07:16:17 AM test_acc 99.800240
    # 300 Epoch
    # 08/25 01:57:31 AM test_acc 100.000000

    
    