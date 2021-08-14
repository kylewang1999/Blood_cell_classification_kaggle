# Setting up working dir
echo Start Copying
SECONDS=0
cp -r /k5wang-volume/Blood_cell_classification_kaggle/kaggle /local/kaggle
cd /k5wang-volume/Blood_cell_classification_kaggle/darts-official
pwd
echo Copying DONE.
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

python train_search.py --batch_size 3 --epochs 50 --save official-BC-50Epoch --learning_rate 0.005

python train_search.py --batch_size 3 --epochs 50 --save FOO --learning_rate 0.005 dataset_path ../kaggle/blood_cell/

