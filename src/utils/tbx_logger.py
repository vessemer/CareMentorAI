from tensorboardX import SummaryWriter

import cv2
import os

from ..modules import metrics as ms
from ..configs import config


class LoggerTBX:
    def __init__(self, log_dir=None):
        if log_dir is not None:
            log_dir = os.path.join(config.PATHS.LOGDIR, log_dir)
        self.writer = SummaryWriter(log_dir=log_dir)
        self.first_group_keys = ['loss', 'reg_loss', 'clf_loss']
        self.second_group_keys = ['map_all', 'map_pathology']

    def __call__(self, train_metr, valid_metr, datas, i=None):
        self.write_scalars(train_metr, valid_metr, i)
        self.write_images(datas, i)
    
    def write_scalars(self, train_metr, valid_metr, i=None):
        self.writer.add_scalars( 'loss_group', self._group_meters(train_metr, valid_metr, self.first_group_keys), i)
        self.writer.add_scalars( 'metric_group', self._group_meters(train_metr, valid_metr, self.second_group_keys), i)

    def write_images(self, datas, i=None):
        for data in datas:
            annotation, bboxes, scores = ms._extract_meta(data)
            filename = data['pid']
            image = cv2.imread(os.path.join(config.PATHS.IMAGES, filename))
            self.writer.add_image_with_boxes('{}/annot'.format(data['pid']), image, annotation[:, :4], 0, dataformats='HWC')
            self.writer.add_image_with_boxes('{}/preds'.format(data['pid']), image, bboxes[:, :4], i, dataformats='HWC')

    def _group_meters(self, train_metr, valid_metr, keys):
            group = { k + '_train': train_metr[k] for k in keys }
            group.update({ k + '_valid': valid_metr[k] for k in keys })
            return group
