# Setting up working dir
echo Start Copying
SECONDS=0           # Tracking Time Elapsed
cp -r /k5wang-volume/Blood_cell_classification_kaggle/kaggle /local/kaggle
# cd /local/Blood_cell_classification_kaggle/darts-LPT/
cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT
pwd
echo Copying DONE. 
duration=$SECONDS   # Displaying Time Elapsed
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."


# Single GPU
python train_search_ts.py --batch_size 3 --epochs 25 --save GPU1080

# Parallel Processing: 2 GPU
# python ../darts-LPT/train_search_ts.py --unrolled --batch_size 3 --epochs 25 --save 50-LR_0.005-FIXED_LOSS --is_parallel 1, --gpu 0,1