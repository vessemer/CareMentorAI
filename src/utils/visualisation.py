import matplotlib.pyplot as plt
import numpy as np
import cv2
from ..configs import config

LABEL_COLOUR = {
    2: (1., 1., 1.),
    1: (1., .6, .6)
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
    img = np.dstack([(img * config.STD) + config.MEAN] * 3)
    img1, img2 = img.copy(), img.copy()
    for idx, bbox in enumerate(pred['bboxes']):
        img1 = visualize_bbox(img1, bbox, category_id_to_name)
    for idx, bbox in enumerate(annotations['bboxes']):
        img2 = visualize_bbox(img2, bbox, category_id_to_name)
    axes[0].imshow(img1)
    axes[1].imshow(img2)
    plt.show()
