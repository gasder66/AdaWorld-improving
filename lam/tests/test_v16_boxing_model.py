import unittest

import torch

from lam.modules.v16_boxing_model import BoxingObjectLAM


class BoxingObjectLAMTest(unittest.TestCase):
    @staticmethod
    def _batch():
        videos = torch.rand(2, 3, 3, 64, 48)
        masks = torch.zeros(2, 3, 2, 64, 48)
        masks[:, :, 0, 8:24, 4:12] = 1
        masks[:, :, 1, 36:52, 32:40] = 1
        background = 1 - masks.sum(dim=2)
        return {"videos": videos, "masks": masks, "background_masks": background}

    def test_shapes_and_object_specific_idm(self):
        torch.manual_seed(0)
        model = BoxingObjectLAM(state_dim=32, latent_dim=8).eval()
        batch = self._batch()
        with torch.no_grad():
            output = model(batch)
            zero_output = model(batch, ablation="zero")
        self.assertEqual(output["z"].shape, (2, 2, 2, 8))
        self.assertEqual(output["reconstruction"].shape, (2, 2, 3, 64, 48))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.isfinite(output["mask_iou"]))
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

    def test_object_input_modes(self):
        batch = self._batch()
        for mode in ("masked_rgb_mask", "mask_only", "masked_rgb"):
            model = BoxingObjectLAM(
                state_dim=32, latent_dim=8, object_input_mode=mode
            ).eval()
            with torch.no_grad():
                output = model(batch)
            self.assertEqual(output["z"].shape, (2, 2, 2, 8))
            expected_channels = BoxingObjectLAM.OBJECT_INPUT_CHANNELS[mode]
            self.assertEqual(model.object_encoder.net[0].in_channels, expected_channels)

        mask_model = BoxingObjectLAM(
            state_dim=32, latent_dim=8, object_input_mode="mask_only"
        ).eval()
        changed = {key: value.clone() for key, value in batch.items()}
        changed["videos"] = torch.rand_like(changed["videos"])
        with torch.no_grad():
            original = mask_model(batch)
            recolored = mask_model(changed)
        self.assertTrue(torch.allclose(original["z"], recolored["z"], atol=1e-6))

    def test_interaction_fdm_and_opponent_ablation(self):
        torch.manual_seed(1)
        model = BoxingObjectLAM(state_dim=32, latent_dim=8, fdm_type="interaction").eval()
        batch = self._batch()
        with torch.no_grad():
            normal = model(batch)
            masked = model(batch, opponent_ablation="mask_state", target_slot=0)
            shuffled = model(batch, opponent_ablation="shuffle_z", target_slot=0)
        self.assertEqual(normal["predicted_object_states"].shape, (2, 2, 2, 32, 16, 12))
        self.assertTrue(torch.isfinite(normal["loss"]))
        self.assertFalse(
            torch.allclose(
                normal["predicted_object_states"][:, :, 0],
                masked["predicted_object_states"][:, :, 0],
            )
        )

        spatial_model = BoxingObjectLAM(state_dim=32, latent_dim=8, fdm_type="spatial_interaction").eval()
        with torch.no_grad():
            spatial_output = spatial_model(batch)
        self.assertEqual(spatial_output["predicted_object_states"].shape, normal["predicted_object_states"].shape)
        self.assertFalse(
            torch.allclose(
                normal["predicted_object_states"][:, :, 0],
                shuffled["predicted_object_states"][:, :, 0],
            )
        )


if __name__ == "__main__":
    unittest.main()
