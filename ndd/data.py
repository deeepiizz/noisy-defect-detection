"""Dataset loading: MVTec AD categories and a synthetic fallback dataset."""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


@dataclass
class Split:
    train: np.ndarray            # (N, H, W) defect-free images in [0, 1]
    test: np.ndarray             # (M, H, W) mix of good and defective images
    labels: np.ndarray           # (M,) 0 = good, 1 = defective
    masks: np.ndarray            # (M, H, W) pixel-level ground truth, bool


def _load_gray(path, size):
    img = Image.open(path).convert("L").resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def _load_mask(path, size):
    if path is None or not Path(path).exists():
        return np.zeros((size, size), dtype=bool)
    img = Image.open(path).convert("L").resize((size, size), Image.NEAREST)
    return np.asarray(img) > 127


def load_mvtec(root, category, size=256):
    """Load one MVTec AD category, e.g. load_mvtec("data/mvtec", "screw").

    Expects the official layout:
        <root>/<category>/train/good/*.png
        <root>/<category>/test/<defect_type>/*.png
        <root>/<category>/ground_truth/<defect_type>/*_mask.png
    """
    base = Path(root) / category
    if not base.exists():
        raise FileNotFoundError(
            f"{base} not found. Download MVTec AD from "
            "https://www.mvtec.com/company/research/datasets/mvtec-ad"
        )

    train = np.stack([_load_gray(p, size) for p in sorted((base / "train" / "good").glob("*.png"))])

    test, labels, masks = [], [], []
    for defect_dir in sorted((base / "test").iterdir()):
        for p in sorted(defect_dir.glob("*.png")):
            test.append(_load_gray(p, size))
            is_good = defect_dir.name == "good"
            labels.append(0 if is_good else 1)
            mask_path = None if is_good else base / "ground_truth" / defect_dir.name / f"{p.stem}_mask.png"
            masks.append(_load_mask(mask_path, size))

    return Split(train, np.stack(test), np.array(labels), np.stack(masks))


def _textured_background(rng, size):
    """Smooth random texture, roughly like brushed metal or fabric."""
    base = rng.normal(0.5, 0.15, (size, size))
    base = ndimage.gaussian_filter(base, sigma=(1.0, 6.0))
    base = (base - base.min()) / (base.max() - base.min() + 1e-8)
    return 0.3 + 0.4 * base


def _add_defect(rng, img):
    """Draw a scratch, blob, or crack; return the modified image and its mask."""
    size = img.shape[0]
    mask = np.zeros_like(img, dtype=bool)
    kind = rng.choice(["scratch", "blob", "crack"])
    yy, xx = np.mgrid[:size, :size]

    if kind == "blob":
        cy, cx = rng.integers(size // 5, 4 * size // 5, 2)
        r = rng.integers(size // 20, size // 10)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    else:
        n_pts = 2 if kind == "scratch" else 6
        pts = rng.integers(size // 8, 7 * size // 8, (n_pts, 2))
        width = 1.5 if kind == "scratch" else 1.0
        for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
            t = np.linspace(0, 1, size * 2)
            ys, xs = (y0 + t * (y1 - y0)).astype(int), (x0 + t * (x1 - x0)).astype(int)
            mask[ys.clip(0, size - 1), xs.clip(0, size - 1)] = True
        mask = ndimage.binary_dilation(mask, iterations=int(np.ceil(width)))

    shift = rng.choice([-1, 1]) * rng.uniform(0.25, 0.4)
    out = img.copy()
    out[mask] = np.clip(out[mask] + shift, 0, 1)
    return out, mask


def make_synthetic(n_train=60, n_test_good=20, n_test_bad=40, size=128, seed=0):
    """Synthetic textured surfaces with injected defects.

    Lets the whole pipeline run (and be tested) without downloading MVTec.
    """
    rng = np.random.default_rng(seed)
    template = _textured_background(rng, size)

    def sample_good():
        # Small per-sample variation in brightness and texture, like real parts.
        jitter = ndimage.gaussian_filter(rng.normal(0, 0.03, (size, size)), 2)
        return np.clip(template * rng.uniform(0.95, 1.05) + jitter, 0, 1).astype(np.float32)

    train = np.stack([sample_good() for _ in range(n_train)])
    test, labels, masks = [], [], []
    for _ in range(n_test_good):
        test.append(sample_good())
        labels.append(0)
        masks.append(np.zeros((size, size), dtype=bool))
    for _ in range(n_test_bad):
        img, m = _add_defect(rng, sample_good())
        test.append(img.astype(np.float32))
        labels.append(1)
        masks.append(m)

    return Split(train, np.stack(test), np.array(labels), np.stack(masks))
