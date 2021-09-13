conda install pytorch==1.1.0 

echo Start Copying
SECONDS=0

mkdir /local/kaggle
cd /local/kaggle
git clone https://github.com/kylewang1999/BCCD_Dataset.git
pwd
ls

# Organize Data Dir

cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT
pwd
echo Copying DONE.
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."



# Default: --batch_size 4 --layers 12 --epochs 300
python train_custom.py --arch DARTS_LPT_410_50 --save 410-darts-lpt-50


