import torch
import torchvision.datasets
import torch.utils.data as data
import os

from utils.pycocotools.coco import COCO
from utils.pycocotools.cocoeval import COCOeval
from utils.pycocotools import mask as COCOmask

from data import VOCroot, COCOroot, VOC_300, VOC_512, COCO_300, COCO_512, COCO_mobile_300, AnnotationTransform, \
    COCODetection, VOCDetection, detection_collate, BaseTransform, preproc

train_sets = [('2014', 'valminusminival')]


# dataset = COCODetection(COCOroot, train_sets, preproc( 300, (104, 117, 123), 0.6))
# epoch_size = len(dataset)
# iiter = data.DataLoader(dataset, 32,shuffle=True, num_workers=2, collate_fn=detection_collate)

dirname = "/media/trans/mnt/data/coco/"
annodir = "annotations"
annofile = "instances_valminusminival2014.json"
_COCO = COCO(os.path.join(dirname, annodir, annofile))

catids = _COCO.getCatIds()
imgids = _COCO.getImgIds()
annids = _COCO.getAnnIds(imgids[0])

cats = _COCO.loadCats(_COCO.getCatIds())
print("# ------------------------------------- # ")
# print(cats)
print(len(cats))
print("# ------------------------------------- # ")
print("catids：", catids)
print("# ------------------------------------- # ")
print("annids: ", annids)
print("# ------------------------------------- # ")
print("imgids: ", imgids[:30])

# ids2 = list()
# ids = list(range(100))
# for t in cats:
#     ids2.append(t['id'])

# print("@ --------------------------------------@ ")
# for i in range(91):
#     if i not in ids2:
#         print(i)
