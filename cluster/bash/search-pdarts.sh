# Setting up working dir
echo Start Copying
SECONDS=0
cp -r /k5wang-volume/Blood_cell_classification_kaggle/kaggle /local/kaggle
cd /k5wang-volume/Blood_cell_classification_kaggle/pdarts-LPT
pwd
echo Copying DONE.
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

python train_search_ts.py --batch_size 8 --epochs 50 --note pdarts-BC-50Epoch --learning_rate 0.005


# python train_search_ts.py --batch_size 8 --epochs 50 --save FOO --note pdarts-BC-50Epoch --learning_rate 0.005


