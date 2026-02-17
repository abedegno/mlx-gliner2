"""
DeBERTa-v2 encoder implemented in Apple MLX.

Ports the disentangled self-attention mechanism from
microsoft/deberta-v3-base for efficient Apple Silicon inference.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import math


@dataclass
class DebertaV2Config:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_act: str = "gelu"
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 0
    layer_norm_eps: float = 1e-7
    vocab_size: int = 128011
    pad_token_id: int = 0
    position_biased_input: bool = False
    relative_attention: bool = True
    max_relative_positions: int = -1
    position_buckets: int = 256
    share_att_key: bool = True
    pos_att_type: List[str] = field(default_factory=lambda: ["p2c", "c2p"])
    norm_rel_ebd: str = "layer_norm"
    conv_kernel_size: int = 0
    embedding_size: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict) -> "DebertaV2Config":
        import inspect
        valid = {k: v for k, v in d.items() if k in inspect.signature(cls).parameters}
        return cls(**valid)


def make_log_bucket_position(
    relative_pos: mx.array,
    bucket_size: int,
    max_position: int,
) -> mx.array:
    sign = mx.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = mx.where(
        (relative_pos < mid) & (relative_pos > -mid),
        mx.array(mid - 1, dtype=relative_pos.dtype),
        mx.abs(relative_pos),
    )
    log_pos = (
        mx.ceil(
            mx.log(abs_pos.astype(mx.float32) / mid)
            / math.log((max_position - 1) / mid)
            * (mid - 1)
        )
        + mid
    )
    bucket_pos = mx.where(
        abs_pos <= mid,
        relative_pos.astype(mx.float32),
        log_pos * sign.astype(mx.float32),
    )
    return bucket_pos


def build_relative_position(
    query_size: int,
    key_size: int,
    bucket_size: int = -1,
    max_position: int = -1,
) -> mx.array:
    q_ids = mx.arange(query_size)
    k_ids = mx.arange(key_size)
    rel_pos_ids = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos_ids = make_log_bucket_position(rel_pos_ids, bucket_size, max_position)
    rel_pos_ids = rel_pos_ids.astype(mx.int32)
    rel_pos_ids = rel_pos_ids[:query_size, :]
    return rel_pos_ids[None, :]


class DisentangledSelfAttention(nn.Module):
    """Disentangled self-attention with content and position components."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) must be divisible by "
                f"num_attention_heads ({config.num_attention_heads})"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.key_proj = nn.Linear(config.hidden_size, self.all_head_size)
        self.value_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.share_att_key = getattr(config, "share_att_key", False)
        self.pos_att_type = config.pos_att_type if config.pos_att_type is not None else []
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            self.position_buckets = getattr(config, "position_buckets", -1)
            self.max_relative_positions = getattr(config, "max_relative_positions", -1)
            if self.max_relative_positions < 1:
                self.max_relative_positions = config.max_position_embeddings
            self.pos_ebd_size = self.max_relative_positions
            if self.position_buckets > 0:
                self.pos_ebd_size = self.position_buckets

            if not self.share_att_key:
                if "c2p" in self.pos_att_type:
                    self.pos_key_proj = nn.Linear(config.hidden_size, self.all_head_size)
                if "p2c" in self.pos_att_type:
                    self.pos_query_proj = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x: mx.array) -> mx.array:
        """Reshape (B, L, D) -> (B*H, L, D/H)."""
        B = x.shape[0]
        L = x.shape[1]
        x = x.reshape(B, L, self.num_attention_heads, -1)
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(B * self.num_attention_heads, L, -1)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        relative_pos: Optional[mx.array] = None,
        rel_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        query_layer = self.transpose_for_scores(self.query_proj(hidden_states))
        key_layer = self.transpose_for_scores(self.key_proj(hidden_states))
        value_layer = self.transpose_for_scores(self.value_proj(hidden_states))

        scale_factor = 1
        if "c2p" in self.pos_att_type:
            scale_factor += 1
        if "p2c" in self.pos_att_type:
            scale_factor += 1
        scale = math.sqrt(query_layer.shape[-1] * scale_factor)

        attention_scores = (query_layer @ key_layer.transpose(0, 2, 1)) / scale

        if self.relative_attention and rel_embeddings is not None:
            rel_att = self.disentangled_attention_bias(
                query_layer, key_layer, relative_pos, rel_embeddings, scale_factor
            )
            if rel_att is not None:
                attention_scores = attention_scores + rel_att

        BH = attention_scores.shape[0]
        B = BH // self.num_attention_heads
        attention_scores = attention_scores.reshape(
            B, self.num_attention_heads,
            attention_scores.shape[-2], attention_scores.shape[-1]
        )

        attention_scores = mx.where(
            attention_mask.astype(mx.bool_),
            attention_scores,
            mx.array(float("-1e9")),
        )

        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_probs = self.dropout(attention_probs)

        attention_probs_flat = attention_probs.reshape(
            -1, attention_probs.shape[-2], attention_probs.shape[-1]
        )
        context_layer = attention_probs_flat @ value_layer

        context_layer = context_layer.reshape(
            B, self.num_attention_heads, context_layer.shape[-2], context_layer.shape[-1]
        )
        context_layer = context_layer.transpose(0, 2, 1, 3)
        context_layer = context_layer.reshape(
            context_layer.shape[0], context_layer.shape[1], -1
        )

        return context_layer

    def disentangled_attention_bias(
        self,
        query_layer: mx.array,
        key_layer: mx.array,
        relative_pos: Optional[mx.array],
        rel_embeddings: mx.array,
        scale_factor: int,
    ) -> Optional[mx.array]:
        if relative_pos is None:
            q_len = query_layer.shape[1]
            k_len = key_layer.shape[1]
            relative_pos = build_relative_position(
                q_len, k_len,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )

        if relative_pos.ndim == 2:
            relative_pos = relative_pos[None, None, :, :]
        elif relative_pos.ndim == 3:
            relative_pos = relative_pos[:, None, :, :]

        att_span = self.pos_ebd_size
        relative_pos = relative_pos.astype(mx.int32)

        rel_embeddings = rel_embeddings[0: att_span * 2, :][None, :, :]

        BH = query_layer.shape[0]
        repeat_count = BH // self.num_attention_heads

        if self.share_att_key:
            pos_query_layer = self.transpose_for_scores(self.query_proj(rel_embeddings))
            pos_query_layer = mx.tile(pos_query_layer, (repeat_count, 1, 1))
            pos_key_layer = self.transpose_for_scores(self.key_proj(rel_embeddings))
            pos_key_layer = mx.tile(pos_key_layer, (repeat_count, 1, 1))
        else:
            pos_key_layer = None
            pos_query_layer = None
            if "c2p" in self.pos_att_type:
                pos_key_layer = self.transpose_for_scores(self.pos_key_proj(rel_embeddings))
                pos_key_layer = mx.tile(pos_key_layer, (repeat_count, 1, 1))
            if "p2c" in self.pos_att_type:
                pos_query_layer = self.transpose_for_scores(self.pos_query_proj(rel_embeddings))
                pos_query_layer = mx.tile(pos_query_layer, (repeat_count, 1, 1))

        score = mx.zeros_like(query_layer @ key_layer.transpose(0, 2, 1))

        if "c2p" in self.pos_att_type and pos_key_layer is not None:
            scale = math.sqrt(pos_key_layer.shape[-1] * scale_factor)
            c2p_att = query_layer @ pos_key_layer.transpose(0, 2, 1)

            c2p_pos = mx.clip(relative_pos + att_span, 0, att_span * 2 - 1)
            c2p_pos_squeezed = c2p_pos.squeeze(0)

            c2p_pos_expanded = mx.broadcast_to(
                c2p_pos_squeezed,
                (query_layer.shape[0], query_layer.shape[1], relative_pos.shape[-1]),
            )
            c2p_att = mx.take_along_axis(c2p_att, c2p_pos_expanded.astype(mx.int32), axis=-1)
            score = score + c2p_att / scale

        if "p2c" in self.pos_att_type and pos_query_layer is not None:
            scale = math.sqrt(pos_query_layer.shape[-1] * scale_factor)

            q_len = query_layer.shape[1]
            k_len = key_layer.shape[1]
            if k_len != q_len:
                r_pos = build_relative_position(
                    k_len, k_len,
                    bucket_size=self.position_buckets,
                    max_position=self.max_relative_positions,
                )
            else:
                r_pos = relative_pos

            p2c_pos = mx.clip(-r_pos + att_span, 0, att_span * 2 - 1)
            p2c_pos_squeezed = p2c_pos.squeeze(0)

            p2c_att = key_layer @ pos_query_layer.transpose(0, 2, 1)
            p2c_pos_expanded = mx.broadcast_to(
                p2c_pos_squeezed,
                (query_layer.shape[0], key_layer.shape[1], key_layer.shape[1]),
            )
            p2c_att = mx.take_along_axis(p2c_att, p2c_pos_expanded.astype(mx.int32), axis=-1)
            p2c_att = p2c_att.transpose(0, 2, 1)
            score = score + p2c_att / scale

        return score


class DebertaV2SelfOutput(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Attention(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.self_attn = DisentangledSelfAttention(config)
        self.output = DebertaV2SelfOutput(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        relative_pos: Optional[mx.array] = None,
        rel_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        self_output = self.self_attn(
            hidden_states,
            attention_mask,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        attention_output = self.output(self_output, hidden_states)
        return attention_output


class DebertaV2Intermediate(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = nn.gelu(hidden_states)
        return hidden_states


class DebertaV2Output(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(self, hidden_states: mx.array, input_tensor: mx.array) -> mx.array:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class DebertaV2Layer(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.attention = DebertaV2Attention(config)
        self.intermediate = DebertaV2Intermediate(config)
        self.output = DebertaV2Output(config)

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
        relative_pos: Optional[mx.array] = None,
        rel_embeddings: Optional[mx.array] = None,
    ) -> mx.array:
        attention_output = self.attention(
            hidden_states,
            attention_mask,
            relative_pos=relative_pos,
            rel_embeddings=rel_embeddings,
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class DebertaV2Embeddings(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        embedding_size = config.embedding_size if config.embedding_size else config.hidden_size
        self.embedding_size = embedding_size
        self.word_embeddings = nn.Embedding(config.vocab_size, embedding_size)
        self.position_biased_input = config.position_biased_input

        if self.position_biased_input:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, embedding_size)
        else:
            self.position_embeddings = None

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, embedding_size)
        else:
            self.token_type_embeddings = None

        if embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(embedding_size, config.hidden_size, bias=False)
        else:
            self.embed_proj = None

        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        embeddings = self.word_embeddings(input_ids)

        if self.position_biased_input and self.position_embeddings is not None:
            seq_length = input_ids.shape[1]
            position_ids = mx.arange(seq_length)[None, :]
            embeddings = embeddings + self.position_embeddings(position_ids)

        if self.embed_proj is not None:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if attention_mask is not None:
            if attention_mask.ndim == 2:
                mask = attention_mask[:, :, None].astype(embeddings.dtype)
                embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaV2Encoder(nn.Module):
    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.layers = [DebertaV2Layer(config) for _ in range(config.num_hidden_layers)]
        self.relative_attention = getattr(config, "relative_attention", False)

        if self.relative_attention:
            max_rel_pos = getattr(config, "max_relative_positions", -1)
            if max_rel_pos < 1:
                max_rel_pos = config.max_position_embeddings
            self.max_relative_positions = max_rel_pos

            position_buckets = getattr(config, "position_buckets", -1)
            self.position_buckets = position_buckets
            pos_ebd_size = max_rel_pos * 2
            if position_buckets > 0:
                pos_ebd_size = position_buckets * 2

            self.rel_embeddings = nn.Embedding(pos_ebd_size, config.hidden_size)

            norm_rel_ebd = getattr(config, "norm_rel_ebd", "none")
            self.norm_rel_ebd_parts = [x.strip() for x in norm_rel_ebd.lower().split("|")]
            if "layer_norm" in self.norm_rel_ebd_parts:
                self.rel_embeddings_LayerNorm = nn.LayerNorm(
                    config.hidden_size, eps=config.layer_norm_eps
                )

    def get_rel_embedding(self) -> Optional[mx.array]:
        if not self.relative_attention:
            return None
        rel_embeddings = self.rel_embeddings.weight
        if "layer_norm" in self.norm_rel_ebd_parts:
            rel_embeddings = self.rel_embeddings_LayerNorm(rel_embeddings)
        return rel_embeddings

    def get_attention_mask(self, attention_mask: mx.array) -> mx.array:
        if attention_mask.ndim <= 2:
            extended = attention_mask[:, None, None, :]
            attention_mask = extended * attention_mask[:, None, :, None]
        elif attention_mask.ndim == 3:
            attention_mask = attention_mask[:, None, :, :]
        return attention_mask

    def __call__(
        self,
        hidden_states: mx.array,
        attention_mask: mx.array,
    ) -> mx.array:
        attention_mask = self.get_attention_mask(attention_mask)

        relative_pos = None
        if self.relative_attention:
            q_len = hidden_states.shape[1]
            relative_pos = build_relative_position(
                q_len, q_len,
                bucket_size=self.position_buckets,
                max_position=self.max_relative_positions,
            )

        rel_embeddings = self.get_rel_embedding()

        for layer in self.layers:
            hidden_states = layer(
                hidden_states,
                attention_mask,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )

        return hidden_states


class DebertaV2Model(nn.Module):
    """Full DeBERTa-v2 model: embeddings + encoder stack."""

    def __init__(self, config: DebertaV2Config):
        super().__init__()
        self.config = config
        self.embeddings = DebertaV2Embeddings(config)
        self.encoder = DebertaV2Encoder(config)

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: Optional[mx.array] = None,
    ) -> mx.array:
        if attention_mask is None:
            attention_mask = mx.ones(input_ids.shape, dtype=mx.int32)

        embedding_output = self.embeddings(input_ids, attention_mask=attention_mask)
        encoder_output = self.encoder(embedding_output, attention_mask)
        return encoder_output
