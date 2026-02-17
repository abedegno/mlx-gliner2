"""
GLiNER2 custom layers ported to MLX.

Implements CountLSTMv2, DownscaledTransformer, and MLP utilities.
"""

from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
import math


def create_mlp(
    input_dim: int,
    intermediate_dims: List[int],
    output_dim: int,
    dropout: float = 0.1,
    activation: str = "gelu",
    add_layer_norm: bool = False,
) -> nn.Module:
    """Create a multi-layer perceptron with specified dimensions."""
    activation_map = {
        "relu": nn.ReLU,
        "gelu": nn.GELU,
        "tanh": nn.Tanh,
        "silu": nn.SiLU,
    }
    act_cls = activation_map.get(activation, nn.GELU)
    layers_list = []
    in_dim = input_dim
    for dim in intermediate_dims:
        layers_list.append(nn.Linear(in_dim, dim))
        if add_layer_norm:
            layers_list.append(nn.LayerNorm(dim))
        layers_list.append(act_cls())
        if dropout > 0:
            layers_list.append(nn.Dropout(dropout))
        in_dim = dim
    layers_list.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers_list)


class DownscaledTransformer(nn.Module):
    """Small transformer operating at a reduced hidden dimension."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.in_projector = nn.Linear(input_size, hidden_size)

        self.transformer_layers = []
        for _ in range(num_layers):
            self.transformer_layers.append(
                TransformerEncoderLayer(hidden_size, num_heads, hidden_size * 2, dropout)
            )

        self.out_projector = create_mlp(
            input_dim=hidden_size + input_size,
            intermediate_dims=[input_size, input_size],
            output_dim=input_size,
            dropout=0.0,
            activation="relu",
            add_layer_norm=False,
        )

    def __call__(self, x: mx.array) -> mx.array:
        """
        Args:
            x: (L, M, input_size)
        Returns:
            (L, M, input_size)
        """
        original_x = x
        x = self.in_projector(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = mx.concatenate([x, original_x], axis=-1)
        x = self.out_projector(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """Single transformer encoder layer with self-attention and FFN."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def __call__(self, src: mx.array) -> mx.array:
        """
        Args:
            src: (L, M, D) -- sequence length, batch (fields), model dim
        Returns:
            (L, M, D)
        """
        L, M, D = src.shape
        src2 = self._self_attention(src)
        src = self.norm1(src + self.dropout(src2))

        src2 = self.linear2(nn.relu(self.linear1(src)))
        src = self.norm2(src + self.dropout(src2))
        return src

    def _self_attention(self, x: mx.array) -> mx.array:
        L, M, D = x.shape
        x_flat = x.reshape(L * M, D)[None, :, :]

        q = self.q_proj(x_flat).reshape(1, L * M, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = self.k_proj(x_flat).reshape(1, L * M, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = self.v_proj(x_flat).reshape(1, L * M, self.nhead, self.head_dim).transpose(0, 2, 1, 3)

        scale = math.sqrt(self.head_dim)
        scores = (q @ k.transpose(0, 1, 3, 2)) / scale
        attn = mx.softmax(scores, axis=-1)
        out = (attn @ v).transpose(0, 2, 1, 3).reshape(1, L * M, D)
        out = self.out_proj(out)
        return out.reshape(L, M, D)


class CountLSTMv2(nn.Module):
    """
    Count-aware structure embedding using GRU + DownscaledTransformer.

    For each predicted count, unrolls a GRU over positional embeddings
    and refines with a small transformer.
    """

    def __init__(self, hidden_size: int, max_count: int = 20):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_count = max_count
        self.pos_embedding = nn.Embedding(max_count, hidden_size)
        self.gru = nn.GRU(input_size=hidden_size, hidden_size=hidden_size)
        self.transformer = DownscaledTransformer(
            hidden_size,
            hidden_size=128,
            num_heads=4,
            num_layers=2,
            dropout=0.1,
        )

    def __call__(self, pc_emb: mx.array, gold_count_val: int) -> mx.array:
        """
        Args:
            pc_emb: (M, D) field embeddings
            gold_count_val: number of count steps

        Returns:
            (gold_count_val, M, D) count-aware embeddings
        """
        M, D = pc_emb.shape
        gold_count_val = min(gold_count_val, self.max_count)

        count_idx = mx.arange(gold_count_val)
        pos_seq = self.pos_embedding(count_idx)
        pos_seq = mx.broadcast_to(pos_seq[:, None, :], (gold_count_val, M, D))

        h0 = pc_emb[None, :, :]

        gru_outputs = []
        h = h0
        for t in range(gold_count_val):
            inp = pos_seq[t:t+1, :, :]
            inp_2d = inp.reshape(M, D)
            h_2d = h.reshape(M, D)
            new_h = self.gru(inp_2d[None, :, :], h_2d)
            if isinstance(new_h, tuple):
                new_h = new_h[0]
            if new_h.ndim == 3:
                h = new_h[:, -1:, :].reshape(1, M, D)
            else:
                h = new_h.reshape(1, M, D)
            gru_outputs.append(h)

        output = mx.concatenate(gru_outputs, axis=0)

        pc_broadcast = mx.broadcast_to(pc_emb[None, :, :], output.shape)
        result = self.transformer(output + pc_broadcast)
        return result
