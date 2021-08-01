# Setting up working dir
echo Start Copying
SECONDS=0
cp -r /k5wang-volume/Blood_cell_classification_kaggle/kaggle /local/kaggle
cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT
pwd
echo Copying DONE.
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

# Train CIFAR network for 300 epoches
python train_custom.py --batch_size 8 --layers 12 --arch DARTS_CIFAR10_TS_1ST --epochs 300 --save CF10-300 --learning_rate 0.005

