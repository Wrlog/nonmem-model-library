import io
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

#: Below this fraction of coloured pixels, a figure has not drawn its data.
#: The emptiest real figure in the library sits far above it and the
#: reproduced blank-plot bug far below; see tests/test_figures.py.
INK_FLOOR = 0.0015


def coloured_fraction(png: bytes) -> float:
    """Fraction of pixels carrying a data colour rather than grey chrome.

    Chrome -- axes, gridlines, text, muted spaghetti lines -- is grey or
    near-grey and so has almost no chroma. Every series colour has plenty.
    """
    pixels = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=int)
    chroma = pixels.max(axis=2) - pixels.min(axis=2)
    return float((chroma > 25).mean())


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
