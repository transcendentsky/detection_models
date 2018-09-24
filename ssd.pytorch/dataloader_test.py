from __future__ import print_function
from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
from tensorboardX import SummaryWriter


def train():
    # if args.dataset == 'COCO':
    #     if args.dataset_root == VOC_ROOT:
    #         if not os.path.exists(COCO_ROOT):
    #             parser.error('Must specify dataset_root if specifying dataset')
    #         print("WARNING: Using default COCO dataset_root because " +
    #               "--dataset_root was not specified.")
    #         args.dataset_root = COCO_ROOT
    #     cfg = coco
    #     dataset = COCODetection(root=args.dataset_root,
    #                             transform=SSDAugmentation(cfg['min_dim'],
    #                                                       MEANS))
    # elif args.dataset == 'VOC':
    # if args.dataset_root == COCO_ROOT:
    #     parser.error('Must specify dataset if specifying dataset_root')
    cfg = voc
    dataset = VOCDetection(root=VOC_ROOT,
                           transform=SSDAugmentation(cfg['min_dim'],
                                                     MEANS))
    epoch_size = len(dataset) // 32
    print("epoch size: ", epoch_size)

    data_loader = data.DataLoader(dataset, 32,
                                  num_workers=6,
                                  shuffle=True, collate_fn=detection_collate,
                                  pin_memory=True)
    # create batch iterator
    batch_iterator = iter(data_loader)
    epoch = 0
    for iteration in range(0, 22):
        if iteration != 0 and (iteration % epoch_size == 0):
            print("\nepoch: ", epoch)
            epoch += 1

        print("\riteration : %d  \t, epoch: %d" % (iteration, epoch), end=' ')
        # load train data
        images, targets = next(batch_iterator)


if __name__ == '__main__':
    train()
