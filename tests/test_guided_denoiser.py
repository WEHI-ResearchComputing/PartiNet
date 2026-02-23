import os
import tempfile

import numpy as np
import mrcfile
import pytest

from partinet.process_utils.guided_denoiser import denoise, transform, standard_scaler


def _write_temp_mrc(array: np.ndarray, dtype: np.dtype) -> str:
    """Write ``array`` to a temporary .mrc file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".mrc")
    os.close(fd)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(array.astype(dtype))
    return path


@ pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.float32])
def test_denoise_handles_various_dtypes(dtype):
    """The ``denoise`` pipeline should accept 16-bit and 32-bit inputs without
    throwing OpenCV kernel-type errors (issue #41).
    """
    data = (np.random.rand(32, 32) * 255).astype(dtype)
    path = _write_temp_mrc(data, dtype)

    try:
        out = denoise(path)
        assert isinstance(out, np.ndarray)
        # our pipeline always returns 8-bit data
        assert out.dtype == np.uint8
        assert out.shape == data.shape
    finally:
        os.unlink(path)


def test_transform_stable_when_constant():
    """``transform`` should not divide by zero if image has no contrast."""
    arr = np.full((4, 4), 100, dtype=np.float32)
    out = transform(arr)
    assert out.dtype == np.uint8
    assert np.all(out == 0)


def test_standard_scaler_normalises():
    arr = np.arange(25, dtype=np.float32).reshape(5, 5)
    scaled = standard_scaler(arr)
    assert scaled.dtype == np.uint8
    # scaled values should not all be equal
    assert scaled.std() > 0
