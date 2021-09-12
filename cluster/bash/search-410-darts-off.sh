# Set up Kaggle
conda install pytorch==1.1.0 
pip install kaggle
mkdir  ~/.kaggle/
cp /k5wang-volume/Blood_cell_classification_kaggle/kaggle/kaggle.json ~/.kaggle/kaggle.json

echo Start Copying
SECONDS=0
# cp -r /k5wang-volume/Blood_cell_classification_kaggle/kaggle/BCCD_Reorganized /local/kaggle/BCCD_Reorganized
mkdir /local/kaggle
cd /local/kaggle
kaggle datasets download -d kylewang1999/pbc-dataset
unzip pbc-dataset.zip
pwd
ls

# Organize Data Dir
org_dir () {
    mv eosinophil 0_eosinophil
    mv lymphocyte 1_lymphocyte
    mv monocyte 2_monocyte
    mv neutrophil 3_neutrophil
}
cd PBC_dataset_split/PBC_dataset_split/Train
org_dir
cd ../Test
org_dir
cd ../Val
org_dir

cd /k5wang-volume/Blood_cell_classification_kaggle/darts-LPT
pwd
echo Copying DONE.
duration=$SECONDS
echo "$(($duration / 60)) minutes and $(($duration % 60)) seconds elapsed."

python train_search.py --batch_size 4 --epochs 50 --save 410-darts-off
# sudo python train_search.py --batch_size 4 --epochs 50 --save FOO --local_mount 0



