"""Unsupervised anomaly detectors. Both are fit on defect-free images only.

Each detector returns a per-pixel anomaly map; the image-level score is the
max of that map (after light smoothing), which is the convention MVTec uses.
"""
import numpy as np
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors


class TemplateZScore:
    """Classical baseline: per-pixel mean/std template and z-score deviation."""

    def __init__(self, smooth_sigma=2.0):
        self.smooth_sigma = smooth_sigma

    def fit(self, train):
        self.mean_ = train.mean(axis=0)
        # Floor the std so flat regions don't blow up the z-score.
        self.std_ = np.maximum(train.std(axis=0), 0.02)
        return self

    def anomaly_map(self, images):
        z = np.abs(images - self.mean_) / self.std_
        return np.stack([ndimage.gaussian_filter(m, self.smooth_sigma) for m in z])


class PatchKNN:
    """Simplified PatchCore: k-NN distance from each test patch to a memory bank
    of normal patches.

    Uses raw pixel patches so it runs without a GPU. Swapping `_features` for
    pretrained CNN features (e.g. ResNet layer2/3) is the natural upgrade.
    """

    def __init__(self, patch=8, stride=4, k=1, bank_size=20000, seed=0):
        self.patch, self.stride, self.k = patch, stride, k
        self.bank_size, self.seed = bank_size, seed

    def _features(self, img):
        p, s = self.patch, self.stride
        win = np.lib.stride_tricks.sliding_window_view(img, (p, p))[::s, ::s]
        gh, gw = win.shape[:2]
        feats = win.reshape(gh * gw, p * p)
        # Remove per-patch brightness so the match focuses on structure.
        feats = feats - feats.mean(axis=1, keepdims=True)
        return feats, (gh, gw)

    def fit(self, train):
        bank = np.concatenate([self._features(img)[0] for img in train])
        if len(bank) > self.bank_size:
            # Random subsampling; PatchCore uses greedy coreset selection instead.
            idx = np.random.default_rng(self.seed).choice(len(bank), self.bank_size, replace=False)
            bank = bank[idx]
        self.nn_ = NearestNeighbors(n_neighbors=self.k).fit(bank)
        return self

    def anomaly_map(self, images):
        maps = []
        for img in images:
            feats, (gh, gw) = self._features(img)
            dist, _ = self.nn_.kneighbors(feats)
            grid = dist.mean(axis=1).reshape(gh, gw)
            # Upsample the patch grid back to image resolution.
            zoom = (img.shape[0] / gh, img.shape[1] / gw)
            full = ndimage.zoom(grid, zoom, order=1)[: img.shape[0], : img.shape[1]]
            maps.append(ndimage.gaussian_filter(full, 2.0))
        return np.stack(maps)


DETECTORS = {"zscore": TemplateZScore, "patchknn": PatchKNN}


def image_scores(maps):
    return maps.reshape(len(maps), -1).max(axis=1)


def segment(anomaly_map, threshold, min_area=10):
    """Threshold a map into a clean binary defect mask using morphology."""
    mask = anomaly_map > threshold
    mask = ndimage.binary_opening(mask, iterations=1)
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    areas = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.isin(labels, np.flatnonzero(areas >= min_area) + 1)
    return keep
