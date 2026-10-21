"""Image- and pixel-level metrics."""
import numpy as np
from sklearn.metrics import roc_auc_score

from .detectors import image_scores


def image_auroc(labels, maps):
    return float(roc_auc_score(labels, image_scores(maps)))


def pixel_auroc(masks, maps):
    # Only defined if the test set has at least one defective pixel.
    y = masks.ravel()
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, maps.ravel()))
