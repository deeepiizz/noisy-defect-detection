"""Synthetic noise models used to stress-test the detectors."""
import numpy as np


def add_gaussian(images, sigma, seed=0):
    """Additive Gaussian noise with standard deviation `sigma` (images in [0, 1])."""
    if sigma <= 0:
        return images.copy()
    rng = np.random.default_rng(seed)
    return np.clip(images + rng.normal(0, sigma, images.shape), 0, 1).astype(np.float32)


def add_salt_pepper(images, amount, seed=0):
    """Set a fraction `amount` of pixels to 0 or 1 at random."""
    if amount <= 0:
        return images.copy()
    rng = np.random.default_rng(seed)
    out = images.copy()
    r = rng.random(images.shape)
    out[r < amount / 2] = 0.0
    out[(r >= amount / 2) & (r < amount)] = 1.0
    return out


NOISE = {"gaussian": add_gaussian, "saltpepper": add_salt_pepper}
