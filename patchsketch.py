import torch


def reshape_patch_embeddings(flat_embeddings, batch_size, num_patches):
    """Reshape [num_patches * batch_size, dim] into [batch_size, num_patches, dim]."""
    if flat_embeddings.dim() != 2:
        raise ValueError("flat_embeddings must be a 2D tensor")

    expected_rows = batch_size * num_patches
    if flat_embeddings.size(0) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows for {batch_size} images and "
            f"{num_patches} patches, got {flat_embeddings.size(0)}"
        )

    return flat_embeddings.view(num_patches, batch_size, -1).transpose(0, 1).contiguous()


def count_nonzero_rows(sketch, zero_tol=1e-12):
    row_norms = torch.norm(sketch, dim=1)
    return int((row_norms > zero_tol).sum().item())


def shrink_frequent_directions(sketch):
    """Apply one Frequent Directions shrink step to a full sketch."""
    if sketch.dim() != 2:
        raise ValueError("sketch must be a 2D tensor")

    _, singular_values, vh = torch.linalg.svd(sketch, full_matrices=False)
    delta = singular_values[-1].pow(2)
    shrunk_values = torch.sqrt(torch.clamp(singular_values.pow(2) - delta, min=0.0))

    # FD guarantees at least one zero row after shrink; make that explicit.
    shrunk_values[-1] = 0.0
    return shrunk_values.unsqueeze(1) * vh


def frequent_directions(rows, sketch_size, zero_tol=1e-12):
    """Sketch a stream of row vectors into an [sketch_size, dim] FD matrix."""
    if rows.dim() != 2:
        raise ValueError("rows must be a 2D tensor")
    if sketch_size < 1:
        raise ValueError("sketch_size must be at least 1")

    num_rows, dim = rows.shape
    if num_rows == 0:
        raise ValueError("rows must contain at least one embedding")

    sketch = rows.new_zeros((sketch_size, dim))
    filled_rows = 0

    for row in rows:
        sketch[filled_rows] = row
        filled_rows += 1

        if filled_rows == sketch_size:
            sketch = shrink_frequent_directions(sketch)
            filled_rows = count_nonzero_rows(sketch, zero_tol=zero_tol)

    return sketch


def consensus_direction(sketch, zero_tol=1e-12):
    """Return the top right singular vector of the non-zero sketch rows."""
    return sketch_diagnostics(sketch, zero_tol=zero_tol)["direction"]


def sketch_diagnostics(sketch, zero_tol=1e-12):
    """Return consensus-direction diagnostics for a sketch matrix."""
    if sketch.dim() != 2:
        raise ValueError("sketch must be a 2D tensor")

    row_norms = torch.norm(sketch, dim=1)
    nonzero_rows = row_norms > zero_tol
    singular_values = sketch.new_zeros(min(sketch.shape))

    if not nonzero_rows.any():
        direction = sketch.new_zeros(sketch.size(1))
        direction[0] = 1.0
        return {
            "direction": direction,
            "singular_values": singular_values,
            "effective_rank": torch.tensor(0, device=sketch.device),
            "nonzero_rows": torch.tensor(0, device=sketch.device),
            "direction_norm": torch.norm(direction),
        }

    active_sketch = sketch[nonzero_rows]
    _, singular_values, vh = torch.linalg.svd(active_sketch, full_matrices=False)
    padded_singular_values = sketch.new_zeros(min(sketch.shape))
    padded_singular_values[: singular_values.numel()] = singular_values

    if singular_values[0] <= zero_tol:
        fallback = active_sketch[0]
        fallback_norm = torch.norm(fallback)
        if fallback_norm <= zero_tol:
            direction = sketch.new_zeros(sketch.size(1))
            direction[0] = 1.0
        else:
            direction = fallback / fallback_norm
    else:
        direction = vh[0]

    return {
        "direction": direction,
        "singular_values": padded_singular_values,
        "effective_rank": (singular_values > zero_tol).sum(),
        "nonzero_rows": nonzero_rows.sum(),
        "direction_norm": torch.norm(direction),
    }


def select_representative_patches(
    batch_embeddings, sketch_size, selected_patches, return_diagnostics=False
):
    """Select top-k patches per image using FD consensus scoring."""
    if batch_embeddings.dim() != 3:
        raise ValueError("batch_embeddings must be a 3D tensor [batch, patches, dim]")

    _, num_patches, _ = batch_embeddings.shape
    if sketch_size < 1 or sketch_size > num_patches:
        raise ValueError("sketch_size must be within [1, num_patches]")
    if selected_patches < 1 or selected_patches > num_patches:
        raise ValueError("selected_patches must be within [1, num_patches]")

    all_indices = []
    all_scores = []
    all_patch_scores = []
    selection_margins = []
    score_spreads = []
    sketch_singular_values = []
    sketch_effective_ranks = []
    sketch_nonzero_rows = []
    consensus_direction_norms = []

    for image_embeddings in batch_embeddings:
        sketch = frequent_directions(image_embeddings, sketch_size)
        diagnostics = sketch_diagnostics(sketch)
        direction = diagnostics["direction"]
        scores = torch.abs(torch.matmul(image_embeddings, direction))
        top_scores, top_indices = torch.topk(
            scores, k=selected_patches, largest=True, sorted=True
        )
        all_indices.append(top_indices)
        all_scores.append(top_scores)
        if return_diagnostics:
            all_patch_scores.append(scores)
            if selected_patches < num_patches:
                sorted_scores = torch.sort(scores, descending=True).values
                selection_margins.append(
                    sorted_scores[selected_patches - 1] - sorted_scores[selected_patches]
                )
            else:
                selection_margins.append(scores.new_zeros(()))
            score_spreads.append(scores.max() - scores.min())
            sketch_singular_values.append(diagnostics["singular_values"])
            sketch_effective_ranks.append(diagnostics["effective_rank"])
            sketch_nonzero_rows.append(diagnostics["nonzero_rows"])
            consensus_direction_norms.append(diagnostics["direction_norm"])

    all_indices = torch.stack(all_indices, dim=0)
    all_scores = torch.stack(all_scores, dim=0)
    if not return_diagnostics:
        return all_indices, all_scores

    return all_indices, all_scores, {
        "all_scores": torch.stack(all_patch_scores, dim=0),
        "selection_margin": torch.stack(selection_margins, dim=0),
        "score_spread": torch.stack(score_spreads, dim=0),
        "sketch_singular_values": torch.stack(sketch_singular_values, dim=0),
        "sketch_effective_rank": torch.stack(sketch_effective_ranks, dim=0),
        "sketch_nonzero_rows": torch.stack(sketch_nonzero_rows, dim=0),
        "consensus_direction_norm": torch.stack(consensus_direction_norms, dim=0),
    }


def gather_selected_embeddings(batch_embeddings, selected_indices):
    if batch_embeddings.dim() != 3:
        raise ValueError("batch_embeddings must be a 3D tensor [batch, patches, dim]")
    if selected_indices.dim() != 2:
        raise ValueError("selected_indices must be a 2D tensor [batch, selected_patches]")

    gather_index = selected_indices.unsqueeze(-1).expand(-1, -1, batch_embeddings.size(-1))
    return torch.gather(batch_embeddings, dim=1, index=gather_index)


def gather_selected_patches(flat_patches, selected_indices, batch_size, num_patches):
    """Gather selected raw patches from patch-major flattened input."""
    if flat_patches.dim() < 2:
        raise ValueError("flat_patches must include batch and feature dimensions")
    if selected_indices.shape[0] != batch_size:
        raise ValueError("selected_indices batch dimension must match batch_size")
    if flat_patches.size(0) != batch_size * num_patches:
        raise ValueError("flat_patches first dimension must equal batch_size * num_patches")

    patch_shape = flat_patches.shape[1:]
    batch_patches = flat_patches.view(num_patches, batch_size, *patch_shape)
    batch_patches = batch_patches.transpose(0, 1)
    gather_shape = (batch_size, selected_indices.size(1), *([1] * len(patch_shape)))
    gather_index = selected_indices.view(gather_shape).expand(
        batch_size, selected_indices.size(1), *patch_shape
    )
    selected = torch.gather(batch_patches, dim=1, index=gather_index)
    return selected.reshape(batch_size * selected_indices.size(1), *patch_shape)


def invariance_loss(selected_embeddings):
    centered = selected_embeddings - selected_embeddings.mean(dim=1, keepdim=True)
    squared_distances = centered.pow(2).sum(dim=2)
    return squared_distances.mean(dim=1)


def covariance_loss(selected_embeddings):
    num_selected = selected_embeddings.size(1)
    if num_selected < 2:
        raise ValueError("selected_embeddings must contain at least 2 patches per image")

    centered = selected_embeddings - selected_embeddings.mean(dim=1, keepdim=True)
    scale = float(num_selected - 1)

    # ||Cov||_F^2 can be computed from the sample Gram matrix, which avoids materializing
    # a full d x d covariance matrix for every image.
    gram = torch.matmul(centered, centered.transpose(1, 2))
    fro_sq = gram.pow(2).sum(dim=(1, 2)) / (scale * scale)
    diag_sq = (centered.pow(2).sum(dim=1) / scale).pow(2).sum(dim=1)
    return torch.clamp(fro_sq - diag_sq, min=0.0)


def patchsketch_loss(selected_embeddings, cov_weight=1.0):
    inv_per_image = invariance_loss(selected_embeddings)
    cov_per_image = covariance_loss(selected_embeddings)

    inv_loss = inv_per_image.mean()
    cov_loss = cov_per_image.mean()
    total_loss = inv_loss + cov_weight * cov_loss
    return total_loss, inv_loss, cov_loss
