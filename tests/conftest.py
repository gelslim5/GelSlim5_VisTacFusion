"""Small CPU config fixture: the paper model config with both encoders replaced by the
deterministic MockEncoder (no pretrained weights needed) and the synthetic dataset."""
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from vistacfusion.utils.config import merge_configs  # noqa: E402


@pytest.fixture
def cfg():
    c = merge_configs(
        os.path.join(ROOT, "configs", "model", "vistacfusion.yaml"),
        os.path.join(ROOT, "configs", "train", "default.yaml"),
        os.path.join(ROOT, "configs", "data", "default.yaml"),
    )
    c.encoder.checkpoint = None          # -> MockEncoder (768-d, 14x14 tokens)
    c.rgb_encoder.checkpoint = None      # -> MockEncoder (1024-d)
    c.dataset = "synthetic"
    return c
