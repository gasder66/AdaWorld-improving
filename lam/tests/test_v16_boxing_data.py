import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "scripts" / "v16" / "prepare_boxing.py"
SPEC = importlib.util.spec_from_file_location("prepare_boxing_v16", MODULE_PATH)
boxing = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = boxing
SPEC.loader.exec_module(boxing)


class BoxingDataHelpersTest(unittest.TestCase):
    @staticmethod
    def _state(boxes, pixels, arms=(0, 0, 0, 0), scores=(0, 0)):
        fighters = []
        for index, (box, count) in enumerate(zip(boxes, pixels)):
            mask = np.zeros((12, 12), dtype=np.uint8)
            mask.flat[:count] = 1
            fighters.append(
                boxing.FighterObservation(
                    name=boxing.FIGHTER_NAMES[index],
                    mask=mask,
                    bbox_xyxy=box,
                    center_xy=boxing._center_from_bbox(box),
                    ram_bbox_xywh=(box[0], box[1], box[2] - box[0], box[3] - box[1]),
                )
            )
        return boxing.FrameState(
            frame=np.zeros((12, 12, 3), dtype=np.uint8),
            fighters=tuple(fighters),
            ram=np.zeros(128, dtype=np.uint8),
            arm_lengths=arms,
            scores=scores,
        )

    def test_sprite_mask_is_not_a_filled_bbox(self):
        frame = np.zeros((12, 12, 3), dtype=np.uint8)
        frame[:] = (10, 20, 30)
        frame[3:8, 4] = (214, 214, 214)
        frame[6, 4:9] = (214, 214, 214)
        mask = boxing._sprite_mask(frame, (214, 214, 214), (3, 2, 7, 7))
        self.assertEqual(int(mask.sum()), 9)
        self.assertEqual(boxing._bbox_from_mask(mask), (4, 3, 9, 8))
        self.assertLess(int(mask[3:8, 4:9].sum()), 25)

    def test_geometry_helpers_distinguish_overlap_and_separation(self):
        left = (0, 0, 10, 10)
        overlap = (8, 2, 15, 8)
        far = (20, 0, 30, 10)
        self.assertEqual(boxing._intersection_area(left, overlap), 12)
        self.assertEqual(boxing._edge_distance(left, overlap), 0.0)
        self.assertEqual(boxing._intersection_area(left, far), 0)
        self.assertEqual(boxing._edge_distance(left, far), 10.0)

    def test_movement_label_uses_continuous_displacement(self):
        self.assertEqual(boxing._movement_label(0.0, 0.0), 0)
        self.assertEqual(boxing._movement_label(0.0, -2.0), 1)
        self.assertEqual(boxing._movement_label(0.0, 2.0), 2)
        self.assertEqual(boxing._movement_label(-3.0, 1.0), 3)
        self.assertEqual(boxing._movement_label(3.0, 1.0), 4)

    def test_punch_transition_phase_classification(self):
        self.assertEqual(boxing._classify_punch_transition([0, 0], [8, 0]), "punch_onset")
        self.assertEqual(boxing._classify_punch_transition([8, 0], [16, 0]), "punch_extend")
        self.assertEqual(boxing._classify_punch_transition([16, 0], [16, 0]), "punch_hold")
        self.assertEqual(boxing._classify_punch_transition([16, 0], [8, 0]), "punch_retract")
        self.assertEqual(boxing._classify_punch_transition([8, 0], [0, 8]), "punch_switch")
        self.assertEqual(boxing._classify_punch_transition([0, 0], [0, 0]), "movement_only")

    def test_interaction_labels_cover_contact_hit_occlusion_and_recovery(self):
        states = [
            self._state(((0, 0, 4, 4), (6, 0, 10, 4)), (20, 20)),
            self._state(((0, 0, 6, 4), (4, 0, 10, 4)), (12, 20), arms=(8, 0, 0, 0)),
            self._state(((0, 0, 4, 4), (7, 0, 11, 4)), (20, 20), scores=(1, 0)),
        ]
        labels = boxing._interaction_labels(
            states, near_distance=3.0, occlusion_drop_ratio=0.15, neutral_arm_value=0
        )
        self.assertEqual(labels["near_labels"].tolist(), [1, 0, 1])
        self.assertEqual(labels["contact_labels"].tolist(), [0, 1, 0])
        self.assertEqual(labels["occlusion_labels"][:, 0].tolist(), [0, 1, 0])
        self.assertEqual(labels["hit_labels"].tolist(), [0, 1])
        self.assertEqual(labels["hit_actor"].tolist(), [-1, 0])
        self.assertEqual(labels["hit_receiver"].tolist(), [-1, 1])
        self.assertEqual(labels["recovery_labels"].tolist(), [0, 1])
        self.assertEqual(labels["punch_miss_labels"][0, 0], 1)
        self.assertEqual(
            boxing._present_interaction_events(labels),
            {"near", "contact", "punch_miss", "hit", "occlusion", "recovery"},
        )


if __name__ == "__main__":
    unittest.main()
