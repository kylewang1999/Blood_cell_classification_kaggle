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
python train_custom_local.py --batch_size 3 --arch DARTS_TS_BC_25EPOCH --epochs 50 --save SEARCH_TS_WAUX

