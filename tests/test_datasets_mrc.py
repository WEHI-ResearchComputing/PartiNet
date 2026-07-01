import sys, os
import numpy as np
import mrcfile
from pathlib import Path

# ensure local `partinet` package is imported instead of any installed version
sys.path.insert(0, os.getcwd())

from partinet.DynamicDet.utils.datasets import LoadImages


def test_loadimages_reads_mrc(tmp_path):
    """LoadImages should accept an uncompressed MRC and return a 3-channel image.

    The network expects BGR input; raw MRCs are single-channel so the loader
    converts to BGR. This test exercises the path added for issue #45.
    """
    arr = (np.random.rand(16, 16) * 255).astype(np.uint8)
    fname = tmp_path / "micro.mrc"
    with mrcfile.new(fname, overwrite=True) as mrc:
        mrc.set_data(arr.astype(np.float32))

    loader = LoadImages(str(tmp_path))
    assert len(loader) == 1
    path, img, img0, cap = next(iter(loader))
    # CHW, three channels
    assert img.ndim == 3 and img.shape[0] == 3
    # original returned image should also be 3‑channel BGR
    assert img0.ndim == 3 and img0.shape[2] == 3
    assert Path(path).suffix.lower() == ".mrc"


def test_save_filename_helper():
    """`_save_filename` should map .mrc inputs to .jpg output names."""
    from partinet.DynamicDet import detect
    p1 = Path("foo.mrc")
    p2 = Path("bar.png")
    assert detect._save_filename(p1) == "foo.png"
    assert detect._save_filename(p2) == "bar.png"
