"""Training loop for the graph models.

One training example is **one day**: a (115, 10) feature matrix, its adjacency, and
115 labels. Batching therefore stacks days, not rows -- a batch of 64 is a (64, 115,
10) tensor. Days are shuffled within the training window, which is safe because each
day is scored independently; the ordering matters for the *split*, not for gradient
descent within it.

Early stopping watches **validation rank IC, not validation loss**. Binary
cross-entropy rewards calibrated probabilities, but the model is used to rank stocks
within a day, and those two objectives peak at different epochs. Stopping on the
metric you actually care about is worth the extra computation.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import rankdata

from marketgraph.features.tensors import PanelTensors

DEVICE = torch.device("cpu")  # 115-node graphs are far too small for MPS to pay off


def mean_rank_ic(scores: np.ndarray, fwd: np.ndarray, mask: np.ndarray) -> float:
    """Average per-day Spearman correlation between scores and realised returns.

    Spearman is Pearson on ranks, so ranking each day and correlating is both exact
    and far quicker than looping through ``scipy.stats.spearmanr``.
    """
    ics = []
    for i in range(scores.shape[0]):
        m = mask[i]
        if m.sum() < 10:
            continue
        a = rankdata(scores[i][m])
        b = rankdata(fwd[i][m])
        if a.std() == 0 or b.std() == 0:
            continue
        ics.append(np.corrcoef(a, b)[0, 1])
    return float(np.mean(ics)) if ics else 0.0


def _batch(tensors: PanelTensors, adj: np.ndarray, assign: np.ndarray, idx: np.ndarray):
    """Assemble one batch of days as torch tensors on DEVICE."""
    return (
        torch.from_numpy(tensors.X[idx]).to(DEVICE),
        torch.from_numpy(adj[assign[idx]]).to(DEVICE),
        torch.from_numpy(tensors.mask[idx]).float().to(DEVICE),
        torch.from_numpy(tensors.y[idx]).to(DEVICE),
    )


@torch.no_grad()
def predict(model, tensors, adj, assign, idx, batch_size: int = 256) -> np.ndarray:
    """Score a set of days. Returns (len(idx), N)."""
    model.eval()
    out = []
    for start in range(0, len(idx), batch_size):
        chunk = idx[start : start + batch_size]
        x, a, m, _ = _batch(tensors, adj, assign, chunk)
        out.append(model(x, a, m).cpu().numpy())
    return np.concatenate(out)


def train_model(
    model,
    tensors: PanelTensors,
    adj: np.ndarray,
    assign: np.ndarray,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    *,
    epochs: int = 60,
    batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 10,
    seed: int = 0,
) -> tuple[torch.nn.Module, float]:
    """Fit ``model``, early-stopping on validation rank IC. Returns (model, best_ic).

    The best-scoring weights are restored before returning, so the model handed back
    is the one that peaked on validation rather than the one from the final epoch.
    """
    from marketgraph.models.gnn import masked_bce

    torch.manual_seed(seed)
    np.random.seed(seed)

    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    rng = np.random.default_rng(seed)

    best_ic = -np.inf
    best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    stale = 0

    for _epoch in range(epochs):
        model.train()
        order = rng.permutation(train_idx)
        for start in range(0, len(order), batch_size):
            chunk = order[start : start + batch_size]
            x, a, m, y = _batch(tensors, adj, assign, chunk)
            loss = masked_bce(model(x, a, m), y, m)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        scores = predict(model, tensors, adj, assign, val_idx)
        ic = mean_rank_ic(scores, tensors.fwd[val_idx], tensors.mask[val_idx])

        if ic > best_ic:
            best_ic = ic
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model, best_ic
