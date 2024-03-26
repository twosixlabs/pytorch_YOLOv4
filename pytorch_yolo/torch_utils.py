import sys
import os
import time
import math
import torch
import numpy as np
from torch.autograd import Variable

import itertools
import struct  # get_image_size
import imghdr  # get_image_size

from pytorch_yolo import utils


def bbox_ious(boxes1, boxes2, x1y1x2y2=True):
    if x1y1x2y2:
        mx = torch.min(boxes1[0], boxes2[0])
        Mx = torch.max(boxes1[2], boxes2[2])
        my = torch.min(boxes1[1], boxes2[1])
        My = torch.max(boxes1[3], boxes2[3])
        w1 = boxes1[2] - boxes1[0]
        h1 = boxes1[3] - boxes1[1]
        w2 = boxes2[2] - boxes2[0]
        h2 = boxes2[3] - boxes2[1]
    else:
        mx = torch.min(boxes1[0] - boxes1[2] / 2.0, boxes2[0] - boxes2[2] / 2.0)
        Mx = torch.max(boxes1[0] + boxes1[2] / 2.0, boxes2[0] + boxes2[2] / 2.0)
        my = torch.min(boxes1[1] - boxes1[3] / 2.0, boxes2[1] - boxes2[3] / 2.0)
        My = torch.max(boxes1[1] + boxes1[3] / 2.0, boxes2[1] + boxes2[3] / 2.0)
        w1 = boxes1[2]
        h1 = boxes1[3]
        w2 = boxes2[2]
        h2 = boxes2[3]
    uw = Mx - mx
    uh = My - my
    cw = w1 + w2 - uw
    ch = h1 + h2 - uh
    mask = ((cw <= 0) + (ch <= 0) > 0)
    area1 = w1 * h1
    area2 = w2 * h2
    carea = cw * ch
    carea[mask] = 0
    uarea = area1 + area2 - carea
    return carea / uarea


def get_region_boxes(boxes_and_confs, explain_mode=False):

    # print('Getting boxes from boxes and confs ...')

    boxes_list = []
    confs_list = []
    obj_list = []

    for item in boxes_and_confs:
        boxes_list.append(item[0])
        confs_list.append(item[1])
        if explain_mode:
            obj_list.append(item[2])

    # boxes: [batch, num1 + num2 + num3, 1, 4]
    # confs: [batch, num1 + num2 + num3, num_classes]
    boxes = torch.cat(boxes_list, dim=1)
    confs = torch.cat(confs_list, dim=1)
    if explain_mode:
        objs = torch.cat(obj_list, dim=1)
        return [boxes, confs, objs]
    else:
        return [boxes, confs]


def convert2cpu(gpu_matrix):
    return torch.FloatTensor(gpu_matrix.size()).copy_(gpu_matrix)


def convert2cpu_long(gpu_matrix):
    return torch.LongTensor(gpu_matrix.size()).copy_(gpu_matrix)



def do_detect(model, img, conf_thresh, nms_thresh, use_cuda=1):
    model.eval()
    with torch.no_grad():
        t0 = time.time()

        if type(img) == np.ndarray and len(img.shape) == 3:  # cv2 image
            img = torch.from_numpy(img.transpose(2, 0, 1)).float().div(255.0).unsqueeze(0)
        elif type(img) == np.ndarray and len(img.shape) == 4:
            img = torch.from_numpy(img.transpose(0, 3, 1, 2)).float().div(255.0)
        else:
            print("unknow image type")
            exit(-1)

        if use_cuda:
            img = img.cuda()
        img = torch.autograd.Variable(img)

        t1 = time.time()

        output = model(img)

        t2 = time.time()

        print('-----------------------------------')
        print('           Preprocess : %f' % (t1 - t0))
        print('      Model Inference : %f' % (t2 - t1))
        print('-----------------------------------')

        return utils.post_processing(img, conf_thresh, nms_thresh, output)

def post_processing_torch(conf_thresh, nms_thresh, output):
    # [num, 1, 4]
    box_array = output[0][0].cpu()
    # [num, num_classes]
    confs = output[1][0].cpu()

    num_classes = confs.shape[1]

    # [batch, num, 4]
    box_array = box_array[:, 0]

    # [num, num_classes] --> [num]
    max_conf,max_id = torch.max(confs, axis=1)
       
    argwhere = max_conf > conf_thresh
    l_box_array = box_array[argwhere, :]
    l_max_conf = max_conf[argwhere]
    l_max_id = max_id[argwhere]
    l_prob_array = confs[argwhere, :]
    
    bboxes = []
    probs = []
    # nms for each class
    for j in range(num_classes):

        cls_argwhere = l_max_id == j
        ll_box_array = l_box_array[cls_argwhere, :]
        ll_max_conf = l_max_conf[cls_argwhere]
        ll_max_id = l_max_id[cls_argwhere]
        ll_prob_array = l_prob_array[cls_argwhere, :]

        keep = nms_torch(ll_box_array, ll_max_conf, nms_thresh)

        if torch.numel(keep):
            ll_box_array = ll_box_array[keep, :]
            ll_max_conf = ll_max_conf[keep]
            ll_max_id = ll_max_id[keep]
            ll_prob_array = ll_prob_array[keep, :]

            for k in range(ll_box_array.shape[0]):
                bboxes.append(ll_box_array[k, :])
                probs.append(ll_prob_array[k, :])
    
    return torch.vstack(bboxes), torch.vstack(probs)

def nms_torch(boxes, confs, nms_thresh=0.5, min_mode=False):
    # print(boxes.shape)
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)
    order = torch.argsort(confs,descending=True)

    keep = []
    while torch.numel(order) > 0:
        idx_self = order[0]
        idx_other = order[1:]

        keep.append(idx_self)

        xx1 = torch.maximum(x1[idx_self], x1[idx_other])
        yy1 = torch.maximum(y1[idx_self], y1[idx_other])
        xx2 = torch.minimum(x2[idx_self], x2[idx_other])
        yy2 = torch.minimum(y2[idx_self], y2[idx_other])

        w = torch.maximum(torch.tensor(0.0), xx2 - xx1)
        h = torch.maximum(torch.tensor(0.0), yy2 - yy1)
        inter = w * h

        if min_mode:
            over = inter / torch.minimum(areas[order[0]], areas[order[1:]])
        else:
            over = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = torch.where(over <= nms_thresh)[0]
        order = order[inds + 1]
    
    return torch.tensor(keep)
