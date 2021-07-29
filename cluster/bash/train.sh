# Setting up working dir
echo Start Copying
SECONDS=0
cp -r /k5wang-volume/Blood_cell_classification_kaggle/kaggle /local/kaggle
cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT
pwd
echo Copying DONE.
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

# python train_custom.py --batch_size 3 --arch DARTS_TS_BC_25EPOCH --epochs 300 --save BC-300Epoch --learning_rate 0.005
python train_custom.py --batch_size 12 --layers 15 --arch DARTS_TS_BC_25EPOCH --epochs 300 --save 25-300-15Layer-12BSize --learning_rate 0.005

# 15 layers could support batch_size 12
#python train_custom.py --batch_size 12 --layers 15 --epochs 300 --save FOO --learning_rate 0.005 --dataset_path ../kaggle/blood_cell/ 

# 108 108 36
# 108 144 36
# 144 144 36
# 144 144 36
# 144 144 36
# 144 144 72
# 144 288 72
# 288 288 72
# 288 288 72
# 288 288 72
# 288 288 144
# 288 576 144
# 576 576 144
# 576 576 144
# 576 576 144
# 576 576 144