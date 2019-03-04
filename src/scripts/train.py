from tqdm import tqdm
from collections import defaultdict
from glob import glob

import numpy as np
import cv2
import pandas as pd
import albumentations
import os 
import pickle

import torch
from torch.utils.data import DataLoader, Dataset
from tensorboardX import SummaryWriter

from ..configs import config
from ..utils import visualisation as vs
from ..utils import tbx_logger as tbx

from ..modules import dataset as ds
from ..modules import metrics as ms
from ..modules import augmentations as augs
from ..modules import learner as lrn
from ..modules import lr_scheduler as lrs

# from retinanet import model as retinanet
# import ..models.retinanet.resnext
# import ..models.retinanet.model
from ..models.retinanet import model as retinanet


CUDA_IDX = 0
torch.cuda.set_device(CUDA_IDX)


labels = pd.read_csv(os.path.join(config.PATHS.CSV, 'labels.csv'))
folds = ds.get_folds(labels, config.PARAMS.NB_FOLDS)


def get_datagens(folds, fold):
    fold = ds.get_fold_split(folds, fold)
    train_dataset = ds.BBoxDataset(labels, fold[0], augmentations=augs.Augmentation())
    valid_dataset = ds.BBoxDataset(labels, fold[1])
    vtrain_dataset = ds.BBoxDataset(labels, fold[0])

    train_datagen = DataLoader(train_dataset, batch_size=config.PARAMS.BATCH_SIZE, shuffle=True, collate_fn=ds.bbox_collater)
    valid_datagen = DataLoader(valid_dataset, batch_size=config.PARAMS.BATCH_SIZE, shuffle=False, collate_fn=ds.bbox_collater)
    vtrain_datagen = DataLoader(vtrain_dataset, batch_size=config.PARAMS.BATCH_SIZE, shuffle=True, collate_fn=ds.bbox_collater)
    return train_datagen, valid_datagen, vtrain_datagen


traced_idxs = [0, 4]
for fold in range(config.PARAMS.NB_FOLDS):
    logger = tbx.LoggerTBX('retinanet34_fold{}'.format(fold))
    train_datagen, valid_datagen, vtrain_datagen = get_datagens(folds, fold)

    focal_loss = retinanet.FocalLoss(iou_lower=.3, iou_upper=.4)
    model = retinanet.resnet34(num_classes=2, focal_loss=focal_loss, pretrained=True, single_channel=True, make_clf=False)

    opt = torch.optim.Adam(model.parameters(), lr=5e-5)#5e-4)
    model = lrn.get_model(model, devices=config.PARAMS.CUDA_DEVICES)

    lr_scheduler = lrs.CosinePiloExt(opt, multiplier=.1, coeff=config.PARAMS.EXP_GAMMA, steps_per_epoch=len(train_datagen) * 128)
    learner = lrn.RetinaLearner(model, opt)

    history = defaultdict(list)
    mval = 0
    for i in range(config.PARAMS.NB_EPOCHS):

        learner.train_on_epoch(train_datagen, lr_scheduler=lr_scheduler)
        lr_scheduler.step()
        el = learner.validate(vtrain_datagen)
        train_metr = ms.estimate_pred(el)
        history['train'].append(train_metr)

        el = learner.validate(valid_datagen)
        valid_metr = ms.estimate_pred(el)
        history['valid'].append(valid_metr)

        logger(train_metr, valid_metr, [el[ti] for ti in traced_idxs], i)
        path = os.path.join(config.PATHS.MODELS, 'retinanet18', 'fold_{}_checkpoint.epoch_{}'.format(fold, '{}'))
        if valid_metr['map_pathology'] > mval:
            mval = valid_metr['map_pathology']
            learner.save(path.format(i))

    learner.save(path.format(i))
    pickle.dump(history, open(path.format('loss'), 'wb'))
