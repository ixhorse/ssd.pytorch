from __future__ import print_function
import sys
import os
import cv2
import json
import argparse 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data

from data import *
from ssd import build_ssd

import pdb

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detection')
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'TT100K'],
                    type=str, help='VOC, COCO, TT100K')
parser.add_argument('--dataset_root', default=VOC_ROOT,
                    help='Dataset root directory path')
parser.add_argument('--trained_model', default='weights/ssd_300_VOC0712.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='test_results/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=True, type=bool,
                    help='Use cuda to train model')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def _visdetection(image, detections, labels):
    detections = np.array(detections).astype(np.int32)
    for box, label in zip(detections, labels):
        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), 100, 2)
        cv2.putText(image, label, (box[0], box[1]), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def test_net(net, cuda, testset, labelmap, transform, thresh):
    """
    """
    # dump predictions and assoc. ground truth to text file for now
    results = dict()
    num_images = len(testset)
    for idx in tqdm(range(num_images)):
        if 'TT100K' in testset.name:
            img_id = testset.pull_id(idx)
        else:
            img_id, _ = testset.pull_anno(idx)

        img = testset.pull_image(idx)
        # [h, w, c] -> [c, h, w]
        x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))
        if cuda:
            x = x.cuda()
        
        y = net(x)      # forward pass
        detections = y.data
        # scale each detection back up to the image
        scale = np.array([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_box = []
        pred_score = []
        pred_label = []
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] > 0:
                score = detections[0, i, j, 0].cpu().numpy()
                label_name = labelmap[i-1]
                pt = (detections[0, i, j, 1:]).cpu().numpy() * scale
                coords = (pt[0], pt[1], pt[2], pt[3])
                pred_box.append(coords)
                pred_score.append(score)
                pred_label.append(label_name)
                j += 1
        # _visdetection(img, pred_box, pred_label)
        # cv2.waitKey(0)
        results[img_id] = {'pred_box': pred_box,
                           'pred_score': pred_score,
                           'pred_label': pred_label}
    return results


def test():
    # load net
    if args.dataset == 'VOC':
        cfg = voc
        labelmap = VOC_CLASSES
        testset = VOCDetection(args.dataset_root, [('2007', 'test')], None, VOCAnnotationTransform())
    elif args.dataset == 'TT100K':
        cfg = tt100k
        labelmap = TT100K_CLASSES
        testset = TT100KDetection(args.dataset_root, [('test')], None)

    num_classes = cfg['num_classes'] 
    net = build_ssd('test', 300, num_classes) # initialize SSD
    net.load_state_dict(torch.load(args.trained_model))
    net.eval()
    print('Finished loading model!')
    # load data
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    results = test_net(net, args.cuda, testset, labelmap,
             BaseTransform(net.size, (104, 117, 123)),
             thresh=args.visual_threshold)

    with open(args.save_folder + '/results.json', 'w') as f:
        json.dump(results, f, cls=MyEncoder)

if __name__ == '__main__':
    test()
