import numpy as np
from tqdm import tqdm_notebook
from collections import defaultdict


def get_ious(y_true, y_pred):
    ious = []
    for prediction in y_pred:
        max_intersection = 0
        max_union = 1e-9
        for ground_truth in y_true:
            intersection = np.sum(prediction * ground_truth)
            if intersection > max_intersection:
                max_intersection = intersection
                max_union = np.sum(np.maximum(prediction, ground_truth))
        ious.append(max_intersection / max_union)
    return ious


def local_mean_iou(y_true, y_pred):
    p = 0
    num_true, num_pred = len(y_true), len(y_pred)

    ious = get_ious(y_true, y_pred)
    for t in np.arange(0.5, 1.0, 0.05):
        matches = ious > t
        tp = np.count_nonzero(matches)
        fp = num_pred - tp
        fn = num_true - tp
        p += tp / (tp + fp + fn)
    return p / 10


def get_scores(gt_labels, name2predictions):
    name2scores = defaultdict(list)
    for i, gt_label in tqdm_notebook(enumerate(gt_labels), total=len(gt_labels)):
        for name, pred in name2predictions.items():
            score = local_mean_iou(gt_label, pred[i])
            name2scores[name].append(score)
            print("{}: {}: {}, mean: {}".format(i, name, score, np.mean(name2scores[name])))
    return name2scores


def get_labels(labeled_image):
    labels = []
    for i in np.unique(labeled_image):
        if i:
            mask = labeled_image == i
            labels.append(mask)
    return np.array(labels)

