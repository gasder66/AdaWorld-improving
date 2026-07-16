import unittest

import torch

from lam.modules.v16_boxing_model import BoxingObjectLAM


class BoxingObjectLAMTest(unittest.TestCase):
    def test_shapes_and_object_specific_idm(self):
        torch.manual_seed(0)
        model = BoxingObjectLAM(state_dim=32, latent_dim=8).eval()
        videos = torch.rand(2, 3, 3, 64, 48)
        masks = torch.zeros(2, 3, 2, 64, 48)
        masks[:, :, 0, 8:24, 4:12] = 1
        masks[:, :, 1, 36:52, 32:40] = 1
        background = 1 - masks.sum(dim=2)
        batch = {"videos": videos, "masks": masks, "background_masks": background}
        with torch.no_grad():
            output = model(batch)
            zero_output = model(batch, ablation="zero")
        self.assertEqual(output["z"].shape, (2, 2, 2, 8))
        self.assertEqual(output["reconstruction"].shape, (2, 2, 3, 64, 48))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(
            torch.allclose(
                zero_output["predicted_object_states"],
                zero_output["object_states"][:, :-1],
                atol=1e-7,
            )
        )

        changed = {key: value.clone() for key, value in batch.items()}
        changed["videos"][:, :, :, 36:52, 32:40] = torch.rand_like(changed["videos"][:, :, :, 36:52, 32:40])
        with torch.no_grad():
            changed_output = model(changed)
        self.assertTrue(torch.allclose(output["z"][:, :, 0], changed_output["z"][:, :, 0], atol=1e-6))


if __name__ == "__main__":
    unittest.main()
