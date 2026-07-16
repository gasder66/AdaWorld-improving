import unittest

import torch

from scripts.v16.build_boxing_transition_index import classify_interaction, classify_transition


class BoxingTransitionClassificationTest(unittest.TestCase):
    def test_phase_classes(self):
        zero = torch.tensor([0, 0])
        self.assertEqual(classify_transition(zero, torch.tensor([8, 0])), "punch_onset")
        self.assertEqual(classify_transition(torch.tensor([8, 0]), torch.tensor([16, 0])), "punch_extend")
        self.assertEqual(classify_transition(torch.tensor([16, 0]), torch.tensor([16, 0])), "punch_hold")
        self.assertEqual(classify_transition(torch.tensor([16, 0]), torch.tensor([8, 0])), "punch_retract")
        self.assertEqual(classify_transition(torch.tensor([8, 0]), torch.tensor([0, 8])), "punch_switch")
        self.assertEqual(classify_transition(zero, zero), "movement_only")

    def test_interaction_role_and_precedence(self):
        sample = {
            "near_labels": torch.tensor([0, 0]),
            "contact_labels": torch.tensor([1, 1]),
            "punch_miss_labels": torch.tensor([[0, 0]]),
            "hit_actor": torch.tensor([0]),
            "hit_receiver": torch.tensor([1]),
            "occlusion_labels": torch.tensor([[0, 0], [0, 1]]),
            "recovery_labels": torch.tensor([0]),
        }
        player_events, player_primary = classify_interaction(sample, 0, 0)
        enemy_events, enemy_primary = classify_interaction(sample, 0, 1)
        self.assertIn("hit", player_events)
        self.assertEqual(player_primary, "hit")
        self.assertIn("received_hit", enemy_events)
        self.assertIn("occlusion", enemy_events)
        self.assertEqual(enemy_primary, "received_hit")


if __name__ == "__main__":
    unittest.main()
