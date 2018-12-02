"""Adapted from:
    @longcw faster_rcnn_pytorch: https://github.com/longcw/faster_rcnn_pytorch
    @rbgirshick py-faster-rcnn https://github.com/rbgirshick/py-faster-rcnn
    Licensed under The MIT License [see LICENSE for details]
"""

from __future__ import print_function
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT, VOCAnnotationTransform, VOCDetection, BaseTransform
from data import VOC_CLASSES as labelmap
from data.coco_test import *
import torch.utils.data as data
from ssd import build_ssd

import sys
import os
import time
import argparse
import numpy as np
import pickle
import cv2

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


COCOroot = os.path.join("/media/trans/mnt", "data/coco/")

parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Evaluation')
parser.add_argument('--trained_model',
                    default='weights/ssd300_mAP_77.43_v2.pth', type=str,
                    help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='eval/', type=str,
                    help='File path to save results')
parser.add_argument('--confidence_threshold', default=0.01, type=float,
                    help='Detection confidence threshold')
parser.add_argument('--top_k', default=5, type=int,
                    help='Further restrict the number of predictions to parse')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT,
                    help='Location of VOC root directory')
parser.add_argument('--cleanup', default=True, type=str2bool,
                    help='Cleanup and remove results files following eval')

args = parser.parse_args()

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't using \
              CUDA.  Run with --cuda for optimal eval speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

annopath = os.path.join(args.voc_root, 'VOC2007', 'Annotations', '%s.xml')
imgpath = os.path.join(args.voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
imgsetpath = os.path.join(args.voc_root, 'VOC2007', 'ImageSets',
                          'Main', '{:s}.txt')
YEAR = '2007'
devkit_path = args.voc_root + 'VOC' + YEAR
dataset_mean = (104, 117, 123)
set_type = 'test'


class Timer(object):
    """A simple timer."""

    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


def parse_rec(filename):
    """ Parse a PASCAL VOC xml file """
    tree = ET.parse(filename)
    objects = []
    for obj in tree.findall('object'):
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        obj_struct['pose'] = obj.find('pose').text
        obj_struct['truncated'] = int(obj.find('truncated').text)
        obj_struct['difficult'] = int(obj.find('difficult').text)
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        objects.append(obj_struct)

    return objects


def get_output_dir(name, phase):
    """Return the directory where experimental artifacts are placed.
    If the directory does not exist, it is created.
    A canonical path is built using the name from an imdb and a network
    (if not None).
    """
    filedir = os.path.join(name, phase)
    if not os.path.exists(filedir):
        os.makedirs(filedir)
    return filedir


def test_net(save_folder, net, cuda, dataset, transform, top_k,
             im_size=300, thresh=0.05):
    num_images = len(dataset)
    print("[DEBUG] length: ", num_images)
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(len(labelmap) + 1)]

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer(), 'total':Timer() }
    output_dir = get_output_dir('ssd300_coco_120000', set_type)
    det_file = os.path.join(output_dir, 'detections.pkl')

    if False:
        _t['total'].tic()
        for i in range(num_images):
            # print("[DEBUG] print i = ", i)
            im, gt, h, w = dataset.__getitem__(i)

            x = Variable(im.unsqueeze(0))
            if args.cuda:
                x = x.cuda()
                # print("______________________\n", x.size())
            _t['im_detect'].tic()
            detections = net(x).data
            detect_time = _t['im_detect'].toc(average=False)

            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets

            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                        num_images, detect_time), end='\r')

        total_time = _t['total'].toc()
        print("Total time: ", total_time, "\t ms: ", total_time / float(num_images))

    with open(det_file, 'wb') as f:
        pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    dataset.evaluate_detections(all_boxes, output_dir)


#
# def evaluate_detections(box_list, output_dir, dataset):
#     write_voc_results_file(box_list, dataset)
#     do_python_eval(output_dir)


def main(trained_model):
    # load net
    net = build_ssd('test', 300, 80)
    # print(net)
    net = net.cuda()  # initialize SSD
    net.load_state_dict(torch.load(trained_model))
    # resume_ckpt(trained_model,net)
    net.eval()
    print('Finished loading model!')
    # load data
    # dataset = VOCDetection(args.voc_root, [('2007', set_type)],
    #                        BaseTransform(300, dataset_mean),
    #                        VOCAnnotationTransform())
    dataset = COCODetection(root=COCOroot,
                            image_sets=[('2014', 'minival')],
                            preproc=BaseTransform(300, dataset_mean))

    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(args.save_folder, net, args.cuda, dataset,
             BaseTransform(net.size, dataset_mean), args.top_k, 300,
             thresh=args.confidence_threshold)


def resume_ckpt(trained_model, net):
    checkpoint = torch.load(trained_model)
    # print(list(checkpoint.items())[0][0])
    if 'module.' in list(checkpoint.items())[0][0]:
        pretrained_dict = {'.'.join(k.split('.')[1:]): v for k, v in list(checkpoint.items())}
        checkpoint = pretrained_dict
    for k, v in checkpoint.items():
        if 'vgg.0' in k:
            print(k, v)


if __name__ == "__main__":
    for i in range(10):
        pth = "results/DataParallel/mixupCOCO/1002/ssd300_COCO_" + str(i + 150) + ".pth"
        print(pth)
        # modelname = 'weights/lm/ssd300_VOC_0.pth'
        # modelname = 'weights/ssd300_mAP_77.43_v2.pth'
        # modelname = 'weights/mixup/ssd300_VOC_' + str(i+23) + '0.pth'
        iii = i + 150
        modelname = "results/DataParallel/mixup005/1002/ssd300_VOC_" + str(iii) + ".pth"
        print("----------------------------------\n"
              "     EVAL modelname: {}\n"
              "----------------------------------\n".format(modelname))
        main(modelname)

# AP for aeroplane = 0.8207
# AP for bicycle = 0.8568
# AP for bird = 0.7546
# AP for boat = 0.6952
# AP for bottle = 0.5019
# AP for bus = 0.8479
# AP for car = 0.8584
# AP for cat = 0.8734
# AP for chair = 0.6136
# AP for cow = 0.8243
# AP for diningtable = 0.7906
# AP for dog = 0.8566
# AP for horse = 0.8714
# AP for motorbike = 0.8403
# AP for person = 0.7895
# AP for pottedplant = 0.5069
# AP for sheep = 0.7767
# AP for sofa = 0.7894
# AP for train = 0.8623
# AP for tvmonitor = 0.7670
# Mean AP = 0.7749
