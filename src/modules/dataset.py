import os
import cv2
import numpy as np
import pandas as pd
import sklearn.model_selection

import torch
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose

from ..configs import config


img_transform = Compose([
    ToTensor(),
    Normalize(mean=config.MEAN, std=config.STD)
])


def get_folds(labels, n_splits, random_state=42):
    kfolds = sklearn.model_selection.StratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    grouped = labels.groupby('filename')
    grouped = grouped.aggregate(min)
    ksplit = kfolds.split(grouped, grouped.is_normal)
    ksplit = [[grouped.index[split] for split in fold] for fold in ksplit]
    return ksplit


def get_fold_split(folds, fold):
    return folds[fold][0], folds[fold][1]


class BBoxDataset(Dataset):
    def __init__(self, labels, fold, augmentations=None):
        self.transform = augmentations
        self.labels = labels.copy()
        self.fold = fold.copy()
        self.format_bboxes()

    def __getitem__(self, idx):
        key = self.fold[idx]
        meta = self.labels.query('filename==@key')
        image = self.load_image(key)
        bboxes = self.load_annotations(meta)
        label = self.load_labels(meta)

        data = {"image": image, "bboxes": bboxes, 'category_id': label}
        if self.transform is not None:
            augmented = self.transform(data)
            image, bboxes = augmented["image"], np.array(augmented["bboxes"])
            labels = augmented['category_id']

        bboxes = np.concatenate([bboxes, np.expand_dims(label, -1)], axis=1)
        return self.postprocess(image, bboxes, key)

    def postprocess(self, image, bboxes, key):
        return { 
            'image': img_transform(np.expand_dims(image.mean(-1), -1).astype(np.uint8)),
            'bboxes': bboxes.astype(np.int),
            'pid': key,
        }

    def load_image(self, key):
        path = os.path.join(config.PATHS.IMAGES, key)
        return cv2.imread(path)

    def load_annotations(self, meta):
        annotation = meta.bboxes.values.tolist()
        return np.array(annotation)

    def load_labels(self, meta):
        labels = meta.is_normal.values
        return labels.astype(np.int) + 1

    def num_classes(self):
        return 2

    def __len__(self):
        return len(self.fold)

    def format_bboxes(self):
        polygons = self.labels.coords.values
        polygons = [list(map(int, poly.split(' '))) for poly in polygons]
        polygons = np.array(polygons)
        shape = polygons.shape
        polygons = polygons.reshape(shape[0], shape[1] // 2, 2)

        self.labels['bboxes'] = np.stack([
            polygons[:, :, 0].min(axis=1), 
            polygons[:, :, 1].min(axis=1), 
            polygons[:, :, 0].max(axis=1), 
            polygons[:, :, 1].max(axis=1)
        ]).T.tolist()


def bbox_collater(data):
    imgs = [s['image'] for s in data]
    annots = [s['bboxes'] for s in data]
    pids = [s['pid'] for s in data]

    imgs = torch.stack(imgs)

    max_num_annots = max(annot.shape[0] for annot in annots)
    if max_num_annots > 0:
        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = torch.tensor(annot)
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    return {'image': imgs, 'bboxes': annot_padded, 'pid': pids}
