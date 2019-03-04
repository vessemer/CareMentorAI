import numpy as np
from tqdm import tqdm
import torch
from ..configs import config


# helper function to calculate IoU
def iou(box1, box2):
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    area1 = (x12 - x11) * (y12 - y11)
    area2 = (x22 - x21) * (y22 - y21)

    xi1, yi1, xi2, yi2 = max([x11, x21]), max([y11, y21]), min([x12, x22]), min([y12, y22])
    
    if xi2 <= xi1 or yi2 <= yi1:
        return 0
    else:
        intersect = (xi2-xi1) * (yi2-yi1)
        union = area1 + area2 - intersect
        return intersect / union


def map_iou(boxes_true, boxes_pred, scores, thresholds=[0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]):
    """
    This is an adaptation of MAP metric from the implementation:
    https://www.kaggle.com/chenyc15/mean-average-precision-metric

    Mean average precision at differnet intersection over union (IoU) threshold

    input:
        boxes_true: Mx4 numpy array of ground true bounding boxes of one image. 
                    bbox format: (x1, y1, x2, y2)
        boxes_pred: Nx4 numpy array of predicted bounding boxes of one image. 
                    bbox format: (x1, y1, x2, y2)
        scores:     length N numpy array of scores associated with predicted bboxes
        thresholds: IoU shresholds to evaluate mean average precision on
    output: 
        map: mean average precision of the image
    """
    
    # According to the introduction, images with no ground truth bboxes will not be 
    # included in the map score unless there is a false positive detection
        
    # return None if both are empty, don't count the image in final evaluation
    if len(boxes_true) == 0 and len(boxes_pred) == 0:
        return None

    assert boxes_true.shape[1] == 4 or boxes_pred.shape[1] == 4, "boxes should be 2D arrays with shape[1]=4"
    if len(boxes_pred):
        assert len(scores) == len(boxes_pred), "boxes_pred and scores should be same length"
        # sort boxes_pred by scores in decreasing order
        boxes_pred = boxes_pred[np.argsort(scores)[::-1], :]
    
    map_total = 0
    
    # loop over thresholds
    for t in thresholds:
        matched_bt = set()
        tp, fn = 0, 0
        for i, bt in enumerate(boxes_true):
            matched = False
            for j, bp in enumerate(boxes_pred):
                miou = iou(bt, bp)
                if miou >= t and not matched and j not in matched_bt:
                    matched = True
                    tp += 1 # bt is matched for the first time, count as TP
                    matched_bt.add(j)
            if not matched:
                fn += 1 # bt has no match, count as FN
                
        fp = len(boxes_pred) - len(matched_bt) # FP is the bp that not matched to any bt
        m = tp / (tp + fn + fp)
        map_total += m
    
    return map_total / len(thresholds)


def _extract_meta(el, threshold=config.PARAMS.THRESHOLD):
    scores, labels, bboxes = el['nms_out']
    annotation = el['annotation']

    bboxes = np.concatenate([
        bboxes, 
        np.expand_dims(labels, -1),
    ], -1)
    bboxes = bboxes[scores > threshold].astype(np.uint)
    scores = scores[scores > threshold]

    return annotation, bboxes, scores


def estimate_pred(history):
    mious_pathology = list()
    mious_all = list()

    for i, el in enumerate(history):
        annotation, bboxes, scores = _extract_meta(el)
        miou_all = map_iou(annotation[:, :4], bboxes[:, :4], scores, thresholds=[.5])
        mious_all.append(miou_all)

        annotation = annotation[annotation[:, -1] == 0]
        bboxes, scores = bboxes[bboxes[:, -1] == 0], scores[bboxes[:, -1] == 0]
        miou = map_iou(annotation[:, :4], bboxes[:, :4], scores, thresholds=[.4])
        mious_pathology.append(miou)

    mious_all = np.mean([el for el in mious_all if el is not None])
    mious_pathology = np.mean([el for el in mious_pathology if el is not None])

    return { 
        'map_all': mious_all, 
        'map_pathology': mious_pathology,
        'loss': np.mean([float(v['loss']) for v in history]),
        'reg_loss': np.mean([float(v['bbx_reg_loss']) for v in history]),
        'clf_loss': np.mean([float(v['bbx_clf_loss']) for v in history]),        
    }
