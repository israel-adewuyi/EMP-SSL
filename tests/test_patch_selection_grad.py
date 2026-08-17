import unittest

import torch
import torch.nn as nn

from main_patchsketch import select_candidate_embeddings
from patchsketch import gather_selected_patches


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 4, bias=False)
        self.calls = []

    def forward(self, x):
        self.calls.append((x.size(0), torch.is_grad_enabled(), self.training))
        return self.projection(x)


class PatchSelectionGradientTests(unittest.TestCase):
    def test_candidates_use_no_grad_and_only_selected_patches_are_reencoded(self):
        torch.manual_seed(4)
        batch_size = 2
        num_patches = 4
        selected_count = 2
        flat_patches = torch.randn(
            batch_size * num_patches, 3, requires_grad=True
        )
        model = RecordingEncoder()
        model.train()

        _, selected_indices, _, _ = select_candidate_embeddings(
            model,
            flat_patches,
            batch_size=batch_size,
            num_patches=num_patches,
            sketch_size=2,
            selected_patches=selected_count,
        )
        selected_patches = gather_selected_patches(
            flat_patches,
            selected_indices,
            batch_size=batch_size,
            num_patches=num_patches,
        )
        model(selected_patches).pow(2).sum().backward()

        self.assertEqual(
            model.calls,
            [
                (batch_size * num_patches, False, False),
                (batch_size * selected_count, True, True),
            ],
        )
        expected_grad_rows = torch.zeros(
            batch_size * num_patches, dtype=torch.bool
        )
        batch_offsets = torch.arange(batch_size).unsqueeze(1)
        flat_selected_indices = selected_indices.cpu() * batch_size + batch_offsets
        expected_grad_rows[flat_selected_indices.reshape(-1)] = True
        actual_grad_rows = flat_patches.grad.abs().sum(dim=1) > 0
        self.assertTrue(torch.equal(actual_grad_rows, expected_grad_rows))


if __name__ == "__main__":
    unittest.main()
