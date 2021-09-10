import os
import sys
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

from torch.autograd import Variable
from model import NetworkCIFAR as Network
from model import NetworkHybrid as NetworkHybrid
import custom_dataset
import mendely_dataloader as loader
# import custom_dataset_improved 

parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=8, help='batch size')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=12, help='total number of layers')
parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
parser.add_argument('--auxiliary', action='store_false', default=True, help='use auxiliary tower')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.2, help='drop path probability')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='DARTS_CIFAR10_TS_1ST', help='which architecture to use')
# parser.add_argument('--dataset_path', type=str, default='../kaggle/blood_cell/', help='location of the data corpus')
# parser.add_argument('--dataset_path', type=str, default='../kaggle/BCCD_Reorganized/', help='location of the data corpus')
# parser.add_argument('--dataset_path', type=str, default='../kaggle/PBC_dataset/PBC_dataset/wbc_resized/', help='location of the data corpus')
# parser.add_argument('--dataset_path', type=str, default='../BCCD_Dataset/BCCD_410', help='location of the data corpus')
parser.add_argument('--dataset_410', type=int, default=0, help='Whether use BCCD_410 Dataset')
args = parser.parse_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

# CIFAR_CLASSES = 10
NUM_CLASSES = 8

def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  
  print(os.getcwd())
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
  utils.load(model, args.model_path)

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()

  # _, test_transform = utils._data_transforms_cifar10(args)
  # test_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=test_transform)

  # test_queue = torch.utils.data.DataLoader(
  #     test_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

  ### For Loading Original BCCD Dataset
  # # dataset_path = "./kaggle/blood_cell/" # Path for local
  # dataset_path = args.dataset_path
  # train_data, test_data, valid_data = custom_dataset.parse_dataset(dataset_path)
  # _, test_queue = custom_dataset.preprocess_data(train_data, test_data, args.batch_size)

  
  if args.dataset_410 == 1:
    ## For Loading Mendely PBC_dataset
    dataloaders = loader.get_dataloaders(batch_size = args.batch_size, num_workers = 2, 
      data_dir='../BCCD_Dataset/BCCD_410')
  else:
    dataloaders = loader.get_dataloaders(batch_size = args.batch_size, num_workers = 2)

  model.drop_path_prob = args.drop_path_prob
  test_acc, test_obj = infer(dataloaders[-1], model, criterion)
  logging.info('test_acc %f', test_acc)


def infer(test_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  with torch.no_grad():
    
    ### For original BCCD dataset
    # for step, data in enumerate(test_queue):
    #   input = data['image']
    #   target = data['label']
    #   input = input.to("cuda", dtype=torch.float)
    #   target = target.to("cuda", dtype=torch.long) 

    ### For Mendely PBC_dataset
    for step, (input, target) in enumerate(test_queue):
      input = input.to("cuda", dtype=torch.float)
      target = target.to("cuda", dtype=torch.long) 

      logits, _ = model(input)
      loss = criterion(logits, target)

      prec1, prec5 = utils.accuracy(logits, target, topk=(1, 2))
      n = input.size(0)
      objs.update(loss.item(), n)
      top1.update(prec1.item(), n)
      top5.update(prec5.item(), n)

      if step % args.report_freq == 0:
        logging.info('test %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main() 

