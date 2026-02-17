"""
GLiNER2 Extractor model ported to MLX.

Ties together the DeBERTa-v2 encoder, span representation layer,
count-aware modules, and task-specific heads.
"""

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .encoder.deberta_v2 import DebertaV2Config, DebertaV2Model
from .layers import CountLSTMv2, create_mlp
from .span_rep import SpanRepLayer


class Extractor(nn.Module):
    """
    GLiNER2 Extractor model for MLX inference.

    Architecture:
        - DeBERTa-v2 encoder
        - SpanRepLayer (SpanMarkerV0) for span representations
        - Classifier MLP for classification tasks
        - Count prediction MLP
        - CountLSTMv2 for count-aware structure embeddings
    """

    def __init__(
        self,
        encoder_config: DebertaV2Config,
        max_width: int = 8,
        counting_layer: str = "count_lstm_v2",
        token_pooling: str = "first",
    ):
        super().__init__()
        self.max_width = max_width
        self.token_pooling = token_pooling

        self.encoder = DebertaV2Model(encoder_config)
        hidden_size = encoder_config.hidden_size

        self.span_rep = SpanRepLayer(
            span_mode="markerV0",
            hidden_size=hidden_size,
            max_width=max_width,
            dropout=0.1,
        )

        self.classifier = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size * 2],
            output_dim=1,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

        self.count_pred = create_mlp(
            input_dim=hidden_size,
            intermediate_dims=[hidden_size * 2],
            output_dim=20,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

        self.count_embed = CountLSTMv2(hidden_size)

    def encode(
        self,
        input_ids: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        """Run the encoder and return last hidden state."""
        return self.encoder(input_ids, attention_mask)

    def compute_span_rep(self, token_embeddings: mx.array) -> Dict[str, Any]:
        """
        Compute span representations for token embeddings.

        Args:
            token_embeddings: (text_len, hidden) single-sample token embeddings

        Returns:
            Dict with span_rep, spans_idx, and span_mask
        """
        text_length = token_embeddings.shape[0]

        spans_idx = []
        for i in range(text_length):
            for j in range(self.max_width):
                if i + j < text_length:
                    spans_idx.append([i, i + j])
                else:
                    spans_idx.append([0, 0])

        spans_idx_arr = mx.array(spans_idx, dtype=mx.int32)[None, :, :]

        span_mask = mx.array(
            [[1 if (i + j >= text_length) else 0
              for j in range(self.max_width)]
             for i in range(text_length)],
            dtype=mx.bool_,
        ).reshape(1, -1)

        span_rep = self.span_rep(
            token_embeddings[None, :, :],
            spans_idx_arr,
        ).squeeze(0)

        return {
            "span_rep": span_rep,
            "spans_idx": spans_idx_arr,
            "span_mask": span_mask,
        }
