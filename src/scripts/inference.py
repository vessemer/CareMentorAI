from tqdm import tqdm
from collections import defaultdict

import pandas as pd
import os 
from glob import glob
import pickle

import numpy as np
import cv2

import albumentations

import torch
from torch.utils.data import DataLoader, Dataset

from tensorboardX import SummaryWriter


import sys

sys.path.append('..')
from src.configs import config
from src.utils import visualisation as vs

from src.modules import dataset as ds
from src.modules import metrics as ms
from src.modules import augmentations as augs
from src.modules import learner as lrn
from src.modules import lr_scheduler as lrs

# from retinanet import model as retinanet
import src.models.retinanet.resnext
import src.models.retinanet.model
from src.models.retinanet import model as retinanet


CUDA_IDX = 0
torch.cuda.set_device(CUDA_IDX)


model_name = config.PATHS.MODEL_NAME
checkpoint = os.path.join(config.PATHS.MODELS, model_name)
paths = os.path.join(config.PATHS.IMAGES, '*.jpg')
fold = [os.path.basename(path) for path in glob(paths)]

dataset = ds.BBoxDataset(fold, labels=None)

model = retinanet.resnet18(num_classes=2, focal_loss=None, pretrained=True, single_channel=True, make_clf=False)
model = lrn.get_model(model, checkpoint=checkpoint, devices=config.PARAMS.CUDA_DEVICES)
learner = lrn.RetinaLearner(model)

datagen = DataLoader(dataset, batch_size=config.PARAMS.BATCH_SIZE, shuffle=False, collate_fn=ds.bbox_collater)

predictions = learner.validate(datagen)

df = pd.DataFrame()
data = list()
for idx, el in enumerate(predictions):
    _, bboxes, scores = _extract_meta(el)
    for i, bbox in enumerate(bboxes):
        data.append({
            'filename': el['pid'], 
            'bboxes': ' '.join(bbox[:-1].astype(np.str)), 
            'is_normal': bbox[-1], 
            'scores': scores[i],
        })
data = pd.DataFrame(data)
formatted = '_'.join(os.path.split(model_name))
path = os.path.join(config.PATHS.CSV, 'checkpoint_{}.csv'.format(formatted))
data.to_csv(path, index=False)
data.head()
