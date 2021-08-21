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

# BC-50-300
# cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/
# python test.py --model_path ./eval-BC-50-300-20210731-025530/weights.pt --arch DARTS_TS_BC_50EPOCH --batch_size 8 --layers 12
    # 08/03 04:57:26 AM test_acc 63.329312
    # 6 Epoch 
    # 08/04 07:13:57 AM test_acc 90.148774
    # 8 Epoch
    # 08/04 07:31:23 AM test_acc 86.811419
    # 9 Epoch
    # 08/04 07:39:42 AM test_acc 84.921592
    # 10 Epoch
    # 08/04 07:46:02 AM test_acc 88.902292
    # 60 Epoch
    # 08/04 02:25:18 PM test_acc 88.620828
    # 300 Epoch
    # 08/12 02:43:15 PM test_acc 88.580619

    
# BC-25-300
# cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/
# python test.py --model_path ./eval-BC-25-300-20210731-025530/weights.pt --arch DARTS_TS_BC_25EPOCH --batch_size 8 --layers 12 

    # 08/04 05:36:11 AM test_acc 54.242059

# python test.py --model_path ./eval-BC-50-300-20210808-070655/weights.pt --arch DARTS_TS_BC_50EPOCH --batch_size 8 --layers 12 

# Official Darts 50 Epoch Search
# python test.py --model_path ./eval-OFF-BC-50-300-20210818-014314/weights.pt --arch DARTS_BC_50EPOCH --batch_size 8 --layers 12 
    # 6 Epoch
    # 08/18 02:43:15 AM test_acc 86.288701
    # 300 Epoch
    # 08/19 01:53:04 PM test_acc 88.379574


# Hybrid Darts-LPT
# python test.py --model_path ./eval-darts-hybrid-20210819-135637/weights.pt --arch DARTS_TS_BC_50EPOCH --batch_size 8 --layers 12 
    # 12 Epoch
    # 08/19 03:47:17 PM test_acc 86.529956