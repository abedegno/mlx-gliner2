"""
Span representation layers ported to MLX.

Implements SpanMarkerV0 and SpanRepLayer from GLiNER's architecture.
"""

import mlx.core as mx
import mlx.nn as nn


def create_projection_layer(
    hidden_size: int,
    dropout: float = 0.4,
    out_size: int = None,
) -> nn.Module:
    """Create a projection MLP: Linear -> ReLU -> Dropout -> Linear."""
    if out_size is None:
        out_size = hidden_size
    intermediate = out_size * 4
    layers = [
        nn.Linear(hidden_size, intermediate),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(intermediate, out_size),
    ]
    return nn.Sequential(*layers)


def extract_elements(sequence: mx.array, indices: mx.array) -> mx.array:
    """
    Gather elements from a sequence using indices.

    Args:
        sequence: (B, L, D)
        indices: (B, K)

    Returns:
        (B, K, D)
    """
    B, L, D = sequence.shape
    K = indices.shape[1]
    indices_clamped = mx.clip(indices, 0, L - 1)
    batch_idx = mx.arange(B)[:, None]
    batch_idx = mx.broadcast_to(batch_idx, (B, K))
    return sequence[batch_idx, indices_clamped]


class SpanMarkerV0(nn.Module):
    """
    Span representation using start/end marker projections.

    Projects start and end positions via separate MLPs, concatenates,
    and projects to produce span embeddings.
    """

    def __init__(self, hidden_size: int, max_width: int, dropout: float = 0.4):
        super().__init__()
        self.max_width = max_width
        self.project_start = create_projection_layer(hidden_size, dropout)
        self.project_end = create_projection_layer(hidden_size, dropout)
        self.out_project = create_projection_layer(hidden_size * 2, dropout, hidden_size)

    def __call__(self, h: mx.array, span_idx: mx.array) -> mx.array:
        """
        Args:
            h: Token representations (B, L, D)
            span_idx: Span indices (B, S, 2) where S = L * max_width

        Returns:
            Span representations (B, L, max_width, D)
        """
        B, L, D = h.shape

        start_rep = self.project_start(h)
        end_rep = self.project_end(h)

        start_span_rep = extract_elements(start_rep, span_idx[:, :, 0])
        end_span_rep = extract_elements(end_rep, span_idx[:, :, 1])

        cat = mx.concatenate([start_span_rep, end_span_rep], axis=-1)
        cat = nn.relu(cat)

        out = self.out_project(cat)
        return out.reshape(B, L, self.max_width, D)


class SpanRepLayer(nn.Module):
    """
    Factory for span representation approaches.

    Currently supports 'markerV0' (the default used by GLiNER2).
    """

    def __init__(
        self,
        hidden_size: int,
        max_width: int,
        span_mode: str = "markerV0",
        dropout: float = 0.1,
    ):
        super().__init__()
        if span_mode == "markerV0":
            self.span_rep_layer = SpanMarkerV0(hidden_size, max_width, dropout=dropout)
        else:
            raise ValueError(f"Unknown span mode: {span_mode}")

    def __call__(self, x: mx.array, span_idx: mx.array) -> mx.array:
        return self.span_rep_layer(x, span_idx)
