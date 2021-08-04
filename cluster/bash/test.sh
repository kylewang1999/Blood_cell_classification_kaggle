cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT

# Testing Blood_cell searched model
python test.py --model_path ./eval-Blood_cell-50Epoch-WAUX-LR0.005/weights.pt --arch DARTS_TS_BC_25EPOCH --batch_size 3

# Testing CIFAR searched model
# python test.py --model_path ./eval-CIFAR-50Epoch-WAUX/weights.pt --arch DARTS_CIFAR10_TS_1ST

# cf100-300
# cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/eval-CF100-300-20210801-012406# pwd/k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/eval-CF100-300-20210801-012406
# python test.py --model_path ./eval-CF100-300-20210801-012406/weights.pt --arch DARTS_CIFAR100_TS_1ST --batch_size 8 --layers 12
    # x_train: 9957 | x_valid: 2487 
    # train_q: 1245 | valid_q: 311 
    # 08/02 10:30:25 AM test 000 1.961308e+00 50.000000 75.000000
    # 08/02 10:30:39 AM test 050 9.419383e-01 75.245098 88.970588
    # 08/02 10:30:52 AM test 100 1.031229e+00 73.143564 87.871287
    # 08/02 10:31:06 AM test 150 1.061691e+00 72.682119 87.417219
    # 08/02 10:31:19 AM test 200 1.034549e+00 73.445274 87.873134
    # 08/02 10:31:33 AM test 250 1.082275e+00 71.862550 87.300797
    # 08/02 10:31:47 AM test 300 1.105146e+00 71.594684 86.835548
    # 08/02 10:31:50 AM test_acc 71.531966

# cf10-300
# cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/
# python test.py --model_path ./eval-CF10-300-20210801-012406/weights.pt --arch DARTS_CIFAR10_TS_1ST --batch_size 8 --layers 12
    # 08/03 05:25:18 AM test 000 1.563469e+00 50.000000 87.500000
    # 08/03 05:25:42 AM test 050 1.207436e+00 64.215686 88.235294
    # 08/03 05:26:06 AM test 100 1.173638e+00 65.099010 89.356436
    # 08/03 05:26:30 AM test 150 1.185931e+00 65.314570 89.735099
    # 08/03 05:26:54 AM test 200 1.169508e+00 64.800995 89.863184
    # 08/03 05:27:18 AM test 250 1.193821e+00 62.998008 89.790837
    # 08/03 05:27:42 AM test 300 1.211819e+00 62.126246 89.867110
    # 08/03 05:27:47 AM test_acc 62.042622

# BC-50-300
# cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/
# python test.py --model_path ./eval-BC-50-300-20210731-025530/weights.pt --arch DARTS_TS_BC_50EPOCH --batch_size 8 --layers 12
    # 08/03 04:52:23 AM test 000 1.364837e+00 75.000000 87.500000
    # 08/03 04:53:12 AM test 050 1.893137e+00 60.784314 67.892157
    # 08/03 04:54:01 AM test 100 1.771136e+00 62.128713 71.163366
    # 08/03 04:54:49 AM test 150 1.694795e+00 63.079470 72.930464
    # 08/03 04:55:38 AM test 200 1.710387e+00 62.997512 72.699005
    # 08/03 04:56:27 AM test 250 1.718551e+00 63.047809 72.559761
    # 08/03 04:57:16 AM test 300 1.696040e+00 63.496678 72.882060
    # 08/03 04:57:26 AM test_acc 63.329312
 
# BC-25-300
# cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT/
# python test.py --model_path ./eval-BC-25-300-20210731-025530/weights.pt --arch DARTS_TS_BC_25EPOCH --batch_size 8 --layers 12
    