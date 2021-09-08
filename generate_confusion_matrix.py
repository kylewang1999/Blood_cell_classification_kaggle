from model import NetworkCIFAR as Network
from mendely_dataloader import get_dataloaders
import torch
import genotypes
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import pylab as pl
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt  

NUM_CLASSES = 8

model = Network(36, NUM_CLASSES, 12, False, eval("genotypes.%s" % 'PC_BLOOD_MEND_40'))
model.load_state_dict(torch.load('eval-EXP-20210905-134722/weights.pt'))

_, _ , test_queue = get_dataloaders(dataset_path, batch_size  = 4)
model.drop_path_prob = 0
dataset_path = '/pranjal-volume/mendely_data_final'

y_test = []  
preds = []
for batch in tqdm(test_queue):
    logits, _ = model(item[0].cuda())
    y_test += logits.argmax(dim=1).detach().cpu().tolist()
    preds += item[1].detach().cpu().tolist()
    
cm = confusion_matrix(y_test, preds)

ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix')
int_to_labels = list(test_queue.dataset.class_to_idx)
ax.xaxis.set_ticklabels(int_to_labels); ax.yaxis.set_ticklabels(int_to_labels);

plt.savefig('confusion_matrix.png')
