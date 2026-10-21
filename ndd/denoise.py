"""SciPy-based denoising filters and image-quality metrics."""
import numpy as np
from scipy import ndimage, signal
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def median(img, size=3):
    return ndimage.median_filter(img, size=size)


def gaussian(img, sigma=1.0):
    return ndimage.gaussian_filter(img, sigma=sigma)


def wiener(img, size=5):
    return np.clip(signal.wiener(img, mysize=size), 0, 1).astype(np.float32)


def identity(img):
    return img


FILTERS = {"none": identity, "median": median, "gaussian": gaussian, "wiener": wiener}


def apply(images, name):
    f = FILTERS[name]
    return np.stack([f(img) for img in images]).astype(np.float32)


def quality(clean, restored):
    """Mean PSNR (capped at 100 dB for identical images) and SSIM over a stack."""
    psnr = np.mean([
        100.0 if np.array_equal(c, r) else peak_signal_noise_ratio(c, r, data_range=1.0)
        for c, r in zip(clean, restored)
    ])
    ssim = np.mean([structural_similarity(c, r, data_range=1.0) for c, r in zip(clean, restored)])
    return float(psnr), float(ssim)
