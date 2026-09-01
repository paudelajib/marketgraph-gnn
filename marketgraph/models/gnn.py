"""Graph neural networks, written from scratch in plain PyTorch.

With 115 nodes the adjacency is a dense 115x115 matrix, so a graph convolution is
literally two matrix multiplies. Importing PyTorch Geometric here would add a heavy,
install-fragile dependency, hide the arithmetic behind ``GCNConv``, and buy nothing:
its sparse scatter kernels only pay off in the tens of thousands of nodes.

Every model takes ``(x, adj, mask)`` and returns one score per node per day:

    x     (B, N, F)  features for B days
    adj   (B, N, N)  normalised adjacency for those days
    mask  (B, N)     True where the stock traded
    ->    (B, N)     higher score = more likely to beat the median

The mask is applied after every layer. Absent stocks are padded with zeros in ``x``,
but ``nn.Linear`` has a bias, so without re-masking a non-existent company would emit
a constant message to its neighbours.

The models form a ladder, each adding exactly one idea:

* ``MLP``          per-node only. **The control.** Identical capacity, optimiser and
                   training loop to the GCN, differing only in that no information
                   crosses between stocks. Any gap between this and the GCN is
                   attributable to the graph and to nothing else -- which is the one
                   comparison this whole project exists to make.
* ``GCN``          averages each node with its neighbours. Fixed edge weights.
* ``GAT``          learns *how much* to weight each neighbour, via attention.
* ``LearnedGCN``   learns the graph itself, with no correlation or sector input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConv(nn.Module):
    """One graph convolution: ``H' = A_hat @ (H W)``.

    That is the whole of message passing. ``H W`` transforms each node's features
    independently; left-multiplying by the normalised adjacency replaces each node's
    vector with a weighted average over itself and its neighbours.
    """

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        return torch.bmm(adj, self.lin(h))


class GraphAttention(nn.Module):
    """Multi-head attention over the graph's existing edges (GAT).

    A GCN weights neighbours by degree alone. Attention instead learns the weight
    from the pair of feature vectors, so a semiconductor stock can attend strongly to
    one peer and weakly to another even when the graph treats those edges alike.

    Attention is restricted to edges present in ``adj``: scores at non-edges are set
    to -inf before the softmax. Letting every node attend to all others would discard
    the graph and rebuild a fully-connected transformer.
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4) -> None:
        super().__init__()
        self.heads = heads
        self.head_dim = out_dim // heads
        self.lin = nn.Linear(in_dim, out_dim)
        self.attn_src = nn.Parameter(torch.empty(heads, self.head_dim))
        self.attn_dst = nn.Parameter(torch.empty(heads, self.head_dim))
        nn.init.xavier_uniform_(self.attn_src)
        nn.init.xavier_uniform_(self.attn_dst)

    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        B, N, _ = h.shape
        z = self.lin(h).view(B, N, self.heads, self.head_dim)

        # Additive attention, decomposed so we never materialise (B,N,N,head_dim).
        src = (z * self.attn_src).sum(-1)  # (B, N, heads)
        dst = (z * self.attn_dst).sum(-1)
        logits = F.leaky_relu(
            src.permute(0, 2, 1).unsqueeze(-1) + dst.permute(0, 2, 1).unsqueeze(-2),
            negative_slope=0.2,
        )  # (B, heads, N, N)

        edges = (adj > 0).unsqueeze(1)
        logits = logits.masked_fill(~edges, float("-inf"))
        # A node with no surviving edges would softmax over all -inf and give NaN.
        alpha = torch.softmax(logits, dim=-1).nan_to_num(0.0)

        out = torch.einsum("bhij,bjhd->bihd", alpha, z)
        return out.reshape(B, N, self.heads * self.head_dim)


class NodeHead(nn.Module):
    """Shared output stack mapping node embeddings to one score each."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim), nn.Dropout(dropout), nn.Linear(dim, 1)
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h).squeeze(-1)


class MLP(nn.Module):
    """Per-node baseline. No message passing -- the control for the ablation."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.head = NodeHead(hidden, dropout)
        self.dropout = dropout

    def forward(self, x, adj, mask):
        h = F.relu(self.l1(x)) * mask.unsqueeze(-1)
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.l2(h)) * mask.unsqueeze(-1)
        return self.head(h)


class GCN(nn.Module):
    """Two graph convolutions over a supplied adjacency."""

    def __init__(self, in_dim: int, hidden: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.g1 = GraphConv(in_dim, hidden)
        self.g2 = GraphConv(hidden, hidden)
        self.head = NodeHead(hidden, dropout)
        self.dropout = dropout

    def forward(self, x, adj, mask):
        m = mask.unsqueeze(-1)
        h = F.relu(self.g1(x, adj)) * m
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.g2(h, adj)) * m
        return self.head(h)


class GAT(nn.Module):
    """Attention-weighted message passing over a supplied adjacency."""

    def __init__(
        self, in_dim: int, hidden: int = 64, heads: int = 4, dropout: float = 0.3
    ) -> None:
        super().__init__()
        self.a1 = GraphAttention(in_dim, hidden, heads)
        self.a2 = GraphAttention(hidden, hidden, heads)
        self.head = NodeHead(hidden, dropout)
        self.dropout = dropout

    def forward(self, x, adj, mask):
        m = mask.unsqueeze(-1)
        h = F.elu(self.a1(x, adj)) * m
        h = F.dropout(h, self.dropout, self.training)
        h = F.elu(self.a2(h, adj)) * m
        return self.head(h)


class LearnedGCN(nn.Module):
    """Learns its own adjacency from node embeddings; ignores the supplied graph.

    Each stock gets two free embedding vectors and the graph is
    ``A = softmax(relu(E_src @ E_dst^T))``, trained end-to-end against the prediction
    loss. Asymmetric by construction, so it can express directional lead-lag effects
    that a correlation matrix cannot.

    The embeddings are per-ticker and therefore only meaningful for a fixed universe
    in a fixed order -- which ours is. The learned matrix is also directly
    inspectable, making it the most interesting variant to visualise.
    """

    def __init__(
        self,
        in_dim: int,
        n_nodes: int,
        hidden: int = 64,
        emb_dim: int = 16,
        top_k: int = 8,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.emb_src = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.emb_dst = nn.Parameter(torch.randn(n_nodes, emb_dim) * 0.1)
        self.top_k = top_k
        self.g1 = GraphConv(in_dim, hidden)
        self.g2 = GraphConv(hidden, hidden)
        self.head = NodeHead(hidden, dropout)
        self.dropout = dropout

    def learned_adjacency(self) -> torch.Tensor:
        scores = F.relu(self.emb_src @ self.emb_dst.t())
        # Keep only the strongest k per row so the learned graph stays as sparse as
        # the correlation graphs it is competing against.
        kth = scores.topk(self.top_k, dim=-1).values[:, -1:]
        scores = scores.masked_fill(scores < kth, float("-inf"))
        return torch.softmax(scores, dim=-1).nan_to_num(0.0)

    def forward(self, x, adj, mask):
        m = mask.unsqueeze(-1)
        a = self.learned_adjacency().expand(x.shape[0], -1, -1)
        h = F.relu(self.g1(x, a)) * m
        h = F.dropout(h, self.dropout, self.training)
        h = F.relu(self.g2(h, a)) * m
        return self.head(h)


def masked_bce(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Binary cross-entropy over valid nodes only."""
    loss = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    return (loss * mask).sum() / mask.sum().clamp(min=1)
