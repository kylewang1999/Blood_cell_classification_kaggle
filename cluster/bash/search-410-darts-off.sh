# Set up Kaggle
conda install pytorch==1.1.0 
# pip install kaggle
# mkdir  ~/.kaggle/
# cp /k5wang-volume/Blood_cell_classification_kaggle/kaggle/kaggle.json ~/.kaggle/kaggle.json

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

python train_search.py --batch_size 4 --epochs 50 --save 410-darts-off
# sudo python train_search.py --batch_size 4 --epochs 50 --save FOO --local_mount 0



