import custom_dataset

import torch
import torch.nn as nn

from utils import AvgrageMeter
import torchvision

import utils


dataset_path = '/pranjal-volume/BCCD_Reorganized/'
bs = 32
train_data, test_data, valid_data = custom_dataset.parse_dataset(dataset_path) # True means using colab
train_queue, valid_queue = custom_dataset.preprocess_data(train_data, valid_data, 64)
_, test_queue = custom_dataset.preprocess_data(train_data, test_data, 128)


# Create Model Here
NUM_CLASSES = 4
model = torchvision.models.resnet50(pretrained=False)
model.fc = nn.Linear(2048, NUM_CLASSES, bias=True)
model = model.cuda()


# Create Optimizer Here
optimizer  = torch.optim.Adam(params= model.parameters() ,lr = 1e-5, weight_decay = 5e-4)

# Create Criterion Here
criterion  = nn.CrossEntropyLoss()

def train(model, optimizer, criterion, train_queue):

    acc_1 = AvgrageMeter()
    acc_2 = AvgrageMeter()
    loss_meter = AvgrageMeter()

    for step, data in enumerate(train_queue):

        optimizer.zero_grad()
        logits = model(data['image'].cuda())
        loss = criterion(logits, data['label'].cuda())

        loss.backward()
        optimizer.step()

        prec1, prec2 = utils.accuracy(logits, data['label'].cuda(), topk=(1, 2))
        acc_1.update(prec1)
        acc_2.update(prec2)
        loss_meter.update(loss.item())

    print(acc_1.avg.item(),acc_2.avg.item(),loss_meter.avg)

def valid(model, criterion, valid_queue):

    acc_1 = AvgrageMeter()
    acc_2 = AvgrageMeter()
    loss_meter = AvgrageMeter()
    model.eval()

    with torch.no_grad():
        for data in valid_queue:

            logits = model(data['image'].cuda())
            loss = criterion(logits, data['label'].cuda())
            prec1, prec2 = utils.accuracy(logits, data['label'].cuda(), topk=(1, 2))
            acc_1.update(prec1)
            acc_2.update(prec2)
            loss_meter.update(loss.item())


    print(acc_1.avg.item(),acc_2.avg.item(),loss_meter.avg)



num_epochs = 3000
for epoch in range(num_epochs):
    print('Epoch:',epoch)
    print('Train:',end = ' ')
    train(model = model, optimizer = optimizer, criterion = criterion, train_queue = train_queue)
    print('Valid:',end = ' ')
    valid(model = model, criterion=criterion, valid_queue = valid_queue)
    if epoch % 10 == 0:
        print('Test:',end = ' ')
        valid(model, criterion, test_queue)


