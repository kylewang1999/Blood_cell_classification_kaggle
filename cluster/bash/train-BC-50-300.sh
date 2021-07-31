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
python train_custom.py --batch_size 8 --layers 12 --arch DARTS_TS_BC_50EPOCH --epochs 300 --save BC-50-300 --learning_rate 0.005

