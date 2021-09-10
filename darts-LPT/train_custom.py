# Trains model from scratch
import os
import os.path
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import genotypes
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from  pathlib import Path

from torch.autograd import Variable
from model import NetworkCIFAR as Network # Trains the network in model.py 
from model import NetworkHybrid as NetworkHybrid
# import custom_dataset
import mendely_dataloader as loader


parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.005, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--epochs', type=int, default=120, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=12, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--auxiliary', action='store_false', default=True, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_CIFAR10_TS_1ST', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--is_cifar100', type=int, default=0)
# parser.add_argument('--dataset_path', type=str, default='/local/kaggle/BCCD_Reorganized/', help='location of the data corpus')
parser.add_argument('--local_mount', type=int, default=1, help='1 use /local on kubectl, 0 use persistent volume')
parser.add_argument('--fine_tune', type=int, default=0, help='0 Train all layer, 1 Fine tune only final layer')

args = parser.parse_args()

if args.fine_tune == 0:
  args.save = 'eval-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
else:
  args.save = 'eval-tune-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

# CIFAR_CLASSES = 10
# CIFAR100_CLASSES = 100
NUM_CLASSES = 8
NUM_CLASSES_410 = 5

def save_checkpoint(state, checkpoint=args.save, filename='checkpoint.pth.tar'):
    filepath = os.path.join(checkpoint, filename)
    torch.save(state, filepath)

def main():
  #Memory Usage
  print("GPU MEM FREE: {} MB".format(utils.get_gpu_memory()))

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  torch.cuda.set_device(args.gpu)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %d' % args.gpu)
  logging.info("args = %s", args)

  genotype = eval("genotypes.%s" % args.arch)
  model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype)
  model = model.cuda()

  # Fine tune for the BCCD_410 dataset
  if args.fine_tune:
    utils.load(model, args.model_path)
    for name, child in model.named_children():
        for x, y in child.named_children():
            print(name,x)
    # model.classifier = nn.Linear
    for param in model.parameters(): # Freez all model weights
      param.requires_grad = False
    for param in model.classifier.parameters(): # Un-freez the final classifier
      param.requires_grad = True

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  optimizer = torch.optim.SGD(
      model.parameters(),
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay
      )

  # Load Dataset
  # dataset_path = args.dataset_path
  # train_data, test_data, valid_data = custom_dataset.parse_dataset(dataset_path) 
  # train_queue, valid_queue = custom_dataset.preprocess_data(train_data, valid_data, args.batch_size)
 

  if args.fine_tune == 1:
    dataloaders = loader.get_dataloaders(batch_size = args.batch_size, num_workers = 2, 
      data_dir='../kaggle/BCCD_Dataset/BCCD_410_split')
    args.report_freq = 10
  else:
    if args.local_mount == 0:
      dataloaders = loader.get_dataloaders(batch_size = args.batch_size, num_workers = 2)
    else:
      dataloaders = loader.get_dataloaders(data_dir='/local/kaggle/PBC_dataset_split/PBC_dataset_split',batch_size = args.batch_size, num_workers = 2)
    

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))
  start_epoch = 0
  if args.resume:
    checkpoint = torch.load(args.resume)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
  
  for epoch in range(start_epoch, args.epochs):
    logging.info('epoch %d lr %e', epoch, scheduler.get_lr()[0])
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    train_acc, train_obj = train(dataloaders[0], model, criterion, optimizer)
    scheduler.step()
    logging.info('train_acc %f', train_acc)

    valid_acc, valid_obj = infer(dataloaders[1], model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))
    save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'scheduler': scheduler.state_dict(),
            'optimizer': optimizer.state_dict()})


def train(train_queue, model, criterion, optimizer):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.train()

  for step, (input, target) in enumerate(train_queue):
    input = input.to("cuda", dtype=torch.float)
    target = target.to("cuda", dtype=torch.long) 
  # for step, data in enumerate(train_queue):
  #   input = data['image']
  #   target = data['label']
  #   input = input.to("cuda", dtype=torch.float)
  #   target = target.to("cuda", dtype=torch.long) 

    optimizer.zero_grad()
    logits, logits_aux = model(input)
    loss = criterion(logits, target)
    if args.auxiliary:
      loss_aux = criterion(logits_aux, target)
      loss += args.auxiliary_weight*loss_aux
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    for step, (input, target) in enumerate(valid_queue):
      input = input.to("cuda", dtype=torch.float)
      target = target.to("cuda", dtype=torch.long) 
    # for step, data in enumerate(valid_queue):
    #     input = data['image']
    #     target = data['label']
    #     input = input.to("cuda", dtype=torch.float)
    #     target = target.to("cuda", dtype=torch.long) 

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

