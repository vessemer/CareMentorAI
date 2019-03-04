import matplotlib.pyplot as plt
import numpy as np
import cv2
from ..configs import config

LABEL_COLOUR = {
    1: (1., 1., 1.),
    0: (1., .6, .6),
    -1: (1., 1., 1.)
}


def visualize_bbox(img, bbox, category_id_to_name, colour=LABEL_COLOUR, thickness=1):
    x_min, y_min, x_max, y_max, class_id = bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=colour[class_id], thickness=thickness)
    return img


def visualize_bboxes(pred, annotations, category_id_to_name):
    _, axes = plt.subplots(ncols=2, figsize=(15, 7))
    img = annotations['image'][..., 0]
    if not isinstance(img, np.ndarray):
        img = annotations['image'].data.numpy().copy()[0]
    if img.dtype != np.uint8:
        img = (img * config.STD) + config.MEAN
    if len(img.shape) == 2:
        img = np.dstack([img] * 3)
    img1, img2 = img.copy(), img.copy()
    for idx, bbox in enumerate(pred['bboxes']):
        img1 = visualize_bbox(img1, bbox, category_id_to_name)
    for idx, bbox in enumerate(annotations['bboxes']):
        img2 = visualize_bbox(img2, bbox, category_id_to_name)
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    plt.show()


def plot_losses(history):
    _, axes = plt.subplots(ncols=2, figsize=(20, 6))
    axes[0].plot([l['loss'] for l in history['train']], label='weighted loss train', alpha=0.8)
    axes[0].plot([l['reg_loss'] for l in history['train']], label='reg loss train', alpha=0.5)
    axes[0].plot([l['clf_loss'] for l in history['train']], label='clf loss train', alpha=0.5)

    axes[0].plot([l['loss'] for l in history['valid']], label='weighted loss val', alpha=0.8)
    axes[0].plot([l['reg_loss'] for l in history['valid']], label='reg loss val', alpha=0.5)
    axes[0].plot([l['clf_loss'] for l in history['valid']], label='clf loss val', alpha=0.5)

    axes[0].set_title('Losses')
    axes[0].legend()
    axes[0].grid()

    axes[1].plot([l['map_all'] for l in history['valid']], label='mAP all val', alpha=0.7)
    axes[1].plot([l['map_pathology'] for l in history['valid']], label='mAP pathology val', alpha=0.7)

    axes[1].plot([l['map_all'] for l in history['train']], label='mAP all train', alpha=0.7)
    axes[1].plot([l['map_pathology'] for l in history['train']], label='mAP pathology train', alpha=0.7)

    axes[1].set_title('Meterics')
    axes[1].legend()
    axes[1].grid()

    plt.show()
