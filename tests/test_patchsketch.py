import unittest

import torch

from patchsketch import (
    covariance_loss,
    gather_selected_embeddings,
    patchsketch_loss,
    reshape_patch_embeddings,
    sketch_diagnostics,
    select_representative_patches,
    shrink_frequent_directions,
)


class PatchSketchTests(unittest.TestCase):
    def test_reshape_patch_embeddings_restores_per_image_order(self):
        flat = torch.tensor(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [10.0, 0.0],
                [20.0, 0.0],
                [100.0, 0.0],
                [200.0, 0.0],
            ]
        )

        reshaped = reshape_patch_embeddings(flat, batch_size=2, num_patches=3)
        expected = torch.tensor(
            [
                [[1.0, 0.0], [10.0, 0.0], [100.0, 0.0]],
                [[2.0, 0.0], [20.0, 0.0], [200.0, 0.0]],
            ]
        )
        self.assertTrue(torch.equal(reshaped, expected))

    def test_shrink_matches_expected_nonnegative_singular_values(self):
        sketch = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        shrunk = shrink_frequent_directions(sketch)
        original_singular_values = torch.linalg.svdvals(sketch)
        shrunk_singular_values = torch.linalg.svdvals(shrunk)

        delta = original_singular_values[-1].pow(2)
        expected_sq = torch.clamp(original_singular_values.pow(2) - delta, min=0.0)

        self.assertTrue(torch.all(shrunk_singular_values.pow(2) >= 0))
        self.assertTrue(
            torch.allclose(
                shrunk_singular_values.pow(2), expected_sq, atol=1e-5, rtol=1e-5
            )
        )

    def test_selection_returns_exact_unique_topk_indices(self):
        torch.manual_seed(0)
        batch_embeddings = torch.randn(4, 8, 6)

        selected_indices, selected_scores = select_representative_patches(
            batch_embeddings, sketch_size=4, selected_patches=3
        )

        self.assertEqual(selected_indices.shape, (4, 3))
        self.assertEqual(selected_scores.shape, (4, 3))

        for row in selected_indices:
            self.assertEqual(torch.unique(row).numel(), 3)

    def test_selection_can_return_diagnostics(self):
        torch.manual_seed(0)
        batch_embeddings = torch.randn(2, 5, 4)

        selected_indices, selected_scores, diagnostics = select_representative_patches(
            batch_embeddings,
            sketch_size=3,
            selected_patches=2,
            return_diagnostics=True,
        )

        self.assertEqual(selected_indices.shape, (2, 2))
        self.assertEqual(selected_scores.shape, (2, 2))
        self.assertEqual(diagnostics["all_scores"].shape, (2, 5))
        self.assertEqual(diagnostics["selection_margin"].shape, (2,))
        self.assertEqual(diagnostics["score_spread"].shape, (2,))
        self.assertEqual(diagnostics["sketch_singular_values"].shape, (2, 3))
        self.assertEqual(diagnostics["sketch_effective_rank"].shape, (2,))
        self.assertEqual(diagnostics["sketch_nonzero_rows"].shape, (2,))
        self.assertEqual(diagnostics["consensus_direction_norm"].shape, (2,))

    def test_sketch_diagnostics_returns_unit_direction(self):
        sketch = torch.tensor(
            [
                [3.0, 0.0, 0.0],
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        )

        diagnostics = sketch_diagnostics(sketch)

        self.assertEqual(diagnostics["singular_values"].shape, (3,))
        self.assertEqual(int(diagnostics["effective_rank"].item()), 2)
        self.assertEqual(int(diagnostics["nonzero_rows"].item()), 2)
        self.assertTrue(torch.isclose(diagnostics["direction"].norm(), torch.tensor(1.0)))

    def test_invariance_loss_is_zero_for_identical_embeddings(self):
        selected = torch.ones(2, 3, 4)
        total_loss, inv_loss, _ = patchsketch_loss(selected, cov_weight=1.0)

        self.assertTrue(torch.isclose(inv_loss, torch.tensor(0.0)))
        self.assertTrue(torch.isclose(total_loss, torch.tensor(0.0)))

    def test_covariance_loss_is_zero_for_diagonal_covariance(self):
        selected = torch.tensor(
            [
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [-1.0, 0.0],
                    [0.0, -1.0],
                ]
            ]
        )

        cov_loss = covariance_loss(selected)
        self.assertTrue(torch.allclose(cov_loss, torch.zeros_like(cov_loss), atol=1e-6))

    def test_patchsketch_loss_is_finite_for_random_inputs(self):
        torch.manual_seed(1)
        selected = torch.randn(3, 5, 7)
        total_loss, inv_loss, cov_loss = patchsketch_loss(selected, cov_weight=1.0)

        self.assertTrue(torch.isfinite(total_loss))
        self.assertTrue(torch.isfinite(inv_loss))
        self.assertTrue(torch.isfinite(cov_loss))

    def test_gathered_selection_routes_gradients_only_to_selected_rows(self):
        batch_embeddings = torch.tensor(
            [
                [[3.0, 0.0], [2.0, 0.0], [0.0, 4.0], [0.0, 1.0]],
                [[0.0, 5.0], [0.0, 2.0], [4.0, 0.0], [1.0, 0.0]],
            ],
            requires_grad=True,
        )
        selected_indices, _ = select_representative_patches(
            batch_embeddings.detach(), sketch_size=2, selected_patches=2
        )

        selected = gather_selected_embeddings(batch_embeddings, selected_indices)
        selected.sum().backward()

        selected_mask = torch.zeros_like(batch_embeddings[..., 0], dtype=torch.bool)
        selected_mask.scatter_(1, selected_indices, True)

        grad_mask = batch_embeddings.grad.abs().sum(dim=2) > 0
        self.assertTrue(torch.equal(grad_mask, selected_mask))


if __name__ == "__main__":
    unittest.main()
