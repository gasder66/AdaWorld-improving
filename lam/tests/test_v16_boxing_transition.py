import unittest

import torch

from scripts.v16.build_boxing_transition_index import classify_transition


class BoxingTransitionClassificationTest(unittest.TestCase):
    def test_phase_classes(self):
        zero = torch.tensor([0, 0])
        self.assertEqual(classify_transition(zero, torch.tensor([8, 0])), "punch_onset")
        self.assertEqual(classify_transition(torch.tensor([8, 0]), torch.tensor([16, 0])), "punch_extend")
        self.assertEqual(classify_transition(torch.tensor([16, 0]), torch.tensor([16, 0])), "punch_hold")
        self.assertEqual(classify_transition(torch.tensor([16, 0]), torch.tensor([8, 0])), "punch_retract")
        self.assertEqual(classify_transition(torch.tensor([8, 0]), torch.tensor([0, 8])), "punch_switch")
        self.assertEqual(classify_transition(zero, zero), "movement_only")


if __name__ == "__main__":
    unittest.main()
