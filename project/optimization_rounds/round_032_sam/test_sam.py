import unittest

import torch

from optimization_rounds.round_032_sam.sam import (
    SAMController,
    capture_torch_rng_state,
    effective_batch_windows,
    manually_unscale_gradients_,
    restore_torch_rng_state,
)


class SAMControllerTests(unittest.TestCase):
    def test_perturbation_has_requested_global_norm_and_restores_exactly(self):
        first = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
        second = torch.nn.Parameter(torch.tensor([0.5]))
        optimizer = torch.optim.AdamW([first, second], lr=1e-3)
        first.grad = torch.tensor([3.0, 4.0])
        second.grad = torch.tensor([12.0])
        originals = [first.detach().clone(), second.detach().clone()]
        controller = SAMController(optimizer, rho=0.05)
        norm = controller.perturb()
        self.assertAlmostEqual(norm, 13.0, places=6)
        delta_norm = torch.linalg.vector_norm(
            torch.cat(
                [
                    (first.detach() - originals[0]).reshape(-1),
                    (second.detach() - originals[1]).reshape(-1),
                ]
            )
        )
        self.assertAlmostEqual(float(delta_norm), 0.05, places=6)
        self.assertTrue(controller.is_perturbed)
        controller.restore()
        self.assertTrue(torch.equal(first.detach(), originals[0]))
        self.assertTrue(torch.equal(second.detach(), originals[1]))
        self.assertFalse(controller.is_perturbed)

    def test_restore_preserves_second_pass_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor([2.0])
        controller = SAMController(optimizer, rho=0.05)
        controller.perturb()
        parameter.grad = torch.tensor([7.0])
        controller.restore()
        self.assertTrue(torch.equal(parameter.grad, torch.tensor([7.0])))
        self.assertTrue(torch.equal(parameter.detach(), torch.tensor([1.0])))

    def test_double_perturb_and_unpaired_restore_fail(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor([1.0])
        controller = SAMController(optimizer)
        controller.perturb()
        with self.assertRaises(RuntimeError):
            controller.perturb()
        controller.restore()
        with self.assertRaises(RuntimeError):
            controller.restore()

    def test_perturbation_does_not_advance_adamw_state(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor([1.0])
        controller = SAMController(optimizer)
        controller.perturb()
        controller.restore()
        self.assertEqual(len(optimizer.state), 0)
        parameter.grad = torch.tensor([2.0])
        optimizer.step()
        self.assertEqual(int(optimizer.state[parameter]["step"]), 1)


class GradientScaleTests(unittest.TestCase):
    def test_manual_unscale_matches_original_gradients(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor([1024.0, -2048.0])
        self.assertTrue(manually_unscale_gradients_(optimizer, 1024.0))
        self.assertTrue(
            torch.equal(parameter.grad, torch.tensor([1.0, -2.0]))
        )

    def test_manual_unscale_reports_nonfinite(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=0.1)
        parameter.grad = torch.tensor([float("inf")])
        self.assertFalse(manually_unscale_gradients_(optimizer, 2.0))


class ReplayProtocolTests(unittest.TestCase):
    def test_effective_batch_windows_include_final_partial_window(self):
        windows = list(effective_batch_windows(range(10), 4))
        self.assertEqual(windows, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]])

    def test_rng_replay_reuses_dropout_mask_and_preserves_trajectory(self):
        torch.manual_seed(2000)
        layer = torch.nn.Dropout(p=0.5)
        value = torch.ones(32)
        before = capture_torch_rng_state()
        first = layer(value)
        after_first = capture_torch_rng_state()
        restore_torch_rng_state(before)
        second = layer(value)
        after_second = capture_torch_rng_state()
        self.assertTrue(torch.equal(first, second))
        self.assertTrue(torch.equal(after_first.cpu, after_second.cpu))


if __name__ == "__main__":
    unittest.main()
