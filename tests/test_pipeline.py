import numpy as np
import pytest

from ndd import data, denoise, evaluate, noise
from ndd.detectors import DETECTORS, segment


@pytest.fixture(scope="module")
def split():
    return data.make_synthetic(n_train=20, n_test_good=8, n_test_bad=12, size=64)


def test_synthetic_shapes(split):
    assert split.train.shape[1:] == split.test.shape[1:] == split.masks.shape[1:]
    assert split.labels.sum() == 12
    assert split.masks[split.labels == 1].any(axis=(1, 2)).all()


def test_noise_stays_in_range(split):
    for fn in noise.NOISE.values():
        out = fn(split.test, 0.2)
        assert out.min() >= 0 and out.max() <= 1


def test_median_filter_beats_saltpepper_noise(split):
    noisy = noise.add_salt_pepper(split.test, 0.1)
    psnr_noisy, _ = denoise.quality(split.test, noisy)
    psnr_med, _ = denoise.quality(split.test, denoise.apply(noisy, "median"))
    assert psnr_med > psnr_noisy


@pytest.mark.parametrize("name", list(DETECTORS))
def test_detectors_separate_clean_data(split, name):
    maps = DETECTORS[name]().fit(split.train).anomaly_map(split.test)
    assert maps.shape == split.test.shape
    assert evaluate.image_auroc(split.labels, maps) > 0.8


def test_segment_removes_small_specks():
    m = np.zeros((32, 32))
    m[5, 5] = 1.0                 # single-pixel speck
    m[15:22, 15:22] = 1.0         # real region
    out = segment(m, 0.5, min_area=10)
    assert not out[5, 5] and out[18, 18]
