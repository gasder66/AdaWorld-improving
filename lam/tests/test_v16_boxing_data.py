import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "v16" / "prepare_boxing.py"
SPEC = importlib.util.spec_from_file_location("prepare_boxing_v16", MODULE_PATH)
boxing = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(boxing)


def test_sprite_mask_is_not_a_filled_bbox():
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    frame[:] = (10, 20, 30)
    frame[3:8, 4] = (214, 214, 214)
    frame[6, 4:9] = (214, 214, 214)
    mask = boxing._sprite_mask(frame, (214, 214, 214), (3, 2, 7, 7))
    assert int(mask.sum()) == 9
    assert boxing._bbox_from_mask(mask) == (4, 3, 9, 8)
    assert int(mask[3:8, 4:9].sum()) < 25


def test_geometry_helpers_distinguish_overlap_and_separation():
    left = (0, 0, 10, 10)
    overlap = (8, 2, 15, 8)
    far = (20, 0, 30, 10)
    assert boxing._intersection_area(left, overlap) == 12
    assert boxing._edge_distance(left, overlap) == 0.0
    assert boxing._intersection_area(left, far) == 0
    assert boxing._edge_distance(left, far) == 10.0


def test_movement_label_uses_continuous_displacement():
    assert boxing._movement_label(0.0, 0.0) == 0
    assert boxing._movement_label(0.0, -2.0) == 1
    assert boxing._movement_label(0.0, 2.0) == 2
    assert boxing._movement_label(-3.0, 1.0) == 3
    assert boxing._movement_label(3.0, 1.0) == 4
