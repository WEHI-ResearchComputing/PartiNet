import numpy as np
import mrcfile
import pytest

from partinet.process_utils.image_io import micrograph_dimensions, load_micrograph_for_detect


def test_micrograph_dimensions_mrc(tmp_path):
    arr = np.random.rand(20, 30).astype(np.float32)
    path = tmp_path / "mic.mrc"
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(arr)
    assert micrograph_dimensions(str(path)) == (30, 20)


def test_load_micrograph_for_detect_mrc(tmp_path):
    arr = (np.random.rand(16, 16) * 100).astype(np.float32)
    path = tmp_path / "mic.mrc"
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(arr)
    img = load_micrograph_for_detect(str(path))
    assert img.ndim == 3 and img.shape[2] == 3
