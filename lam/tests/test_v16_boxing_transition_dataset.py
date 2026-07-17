import tempfile
import unittest
from pathlib import Path

import torch

from lam.datasets.boxing_transition_dataset import BoxingTransitionDataset


class BoxingTransitionDatasetTest(unittest.TestCase):
    def test_temporal_context_and_prediction_horizon_filter_complete_windows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / "episode.pt"
            height, width = 8, 6
            masks = torch.zeros(6, 2, height, width)
            masks[:, 0, 1:3, 1:3] = 1
            masks[:, 1, 5:7, 3:5] = 1
            torch.save(
                {
                    "videos": torch.zeros(6, 3, height, width, dtype=torch.uint8),
                    "masks": masks,
                    "background_masks": 1 - masks.sum(dim=1).clamp(0, 1),
                    "valid_mask": torch.ones(6, 2, dtype=torch.bool),
                    "centers_xy": torch.zeros(6, 2, 2),
                    "arm_lengths": torch.zeros(6, 2, 4),
                },
                sample_path,
            )
            entries = [
                {
                    "path": str(sample_path),
                    "transition": transition,
                    "event": "movement_only",
                    "fighter": "player",
                    "target_slot": 0,
                }
                for transition in range(5)
            ]
            index_path = root / "index.pt"
            torch.save({"entries": entries, "stats": {}}, index_path)

            dataset = BoxingTransitionDataset(
                str(index_path), temporal_context=3, prediction_horizon=2
            )

            self.assertEqual(len(dataset), 2)
            self.assertEqual(dataset[0]["videos"].shape, (5, 3, height, width))
            self.assertEqual(dataset[0]["masks"].shape, (5, 2, height, width))


if __name__ == "__main__":
    unittest.main()
