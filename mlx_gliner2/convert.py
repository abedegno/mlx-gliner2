"""
Convert GLiNER2 PyTorch weights to MLX format.

Downloads from HuggingFace Hub and remaps weight keys to match
the MLX model architecture.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .encoder.deberta_v2 import DebertaV2Config
from .model import Extractor


def download_model(repo_id: str, local_dir: str = None) -> str:
    """Download model files from HuggingFace Hub."""
    from huggingface_hub import snapshot_download

    if local_dir is None:
        local_dir = os.path.join("mlx_models", repo_id.replace("/", "_"))

    snapshot_download(
        repo_id,
        local_dir=local_dir,
        allow_patterns=["*.json", "*.safetensors", "*.bin", "*.txt", "*.model"],
    )
    return local_dir


def load_torch_weights(model_dir: str) -> Dict[str, np.ndarray]:
    """Load weights from safetensors or pytorch bin format."""
    safetensors_path = os.path.join(model_dir, "model.safetensors")
    bin_path = os.path.join(model_dir, "pytorch_model.bin")

    if os.path.exists(safetensors_path):
        from safetensors import safe_open

        weights = {}
        with safe_open(safetensors_path, framework="numpy") as f:
            for key in f.keys():
                weights[key] = f.get_tensor(key)
        return weights
    elif os.path.exists(bin_path):
        import torch

        state_dict = torch.load(bin_path, map_location="cpu")
        return {k: v.numpy() for k, v in state_dict.items()}
    else:
        raise FileNotFoundError(
            f"No model weights found in {model_dir}. "
            f"Expected model.safetensors or pytorch_model.bin"
        )


def remap_key(key: str) -> str:
    """
    Remap a PyTorch weight key to MLX naming conventions.

    Key differences:
    - encoder.encoder.layer.X -> encoder.encoder.layers.X
    - .attention.self. -> .attention.self_attn.
    - encoder.encoder.LayerNorm. -> encoder.encoder.rel_embeddings_LayerNorm.
    - nn.Sequential indices: .N. -> .layers.N.  (for non-encoder parts)
    - transformer.layers -> transformer_layers (for DownscaledTransformer)
    """
    # Encoder layer list: "layer" -> "layers"
    key = key.replace("encoder.encoder.layer.", "encoder.encoder.layers.")

    # Self-attention: "attention.self." -> "attention.self_attn."
    key = key.replace(".attention.self.", ".attention.self_attn.")

    # Relative embedding LayerNorm
    key = key.replace(
        "encoder.encoder.LayerNorm.", "encoder.encoder.rel_embeddings_LayerNorm."
    )

    # DownscaledTransformer: "transformer.layers" -> "transformer_layers"
    # (but NOT encoder.encoder.layers which was already handled)
    key = key.replace(
        "count_embed.transformer.transformer.layers.",
        "count_embed.transformer.transformer_layers.",
    )

    # PyTorch TransformerEncoderLayer stores self_attn.out_proj,
    # our implementation uses out_proj directly on the layer
    key = key.replace(".self_attn.out_proj.", ".out_proj.")

    # nn.Sequential indexing: PyTorch uses .0., .1., etc.
    # MLX nn.Sequential uses .layers.0., .layers.1., etc.
    # Apply to non-encoder parts: classifier, count_pred, span_rep, out_projector
    import re as _re

    sequential_prefixes = [
        "classifier",
        "count_pred",
        "span_rep.span_rep_layer.project_start",
        "span_rep.span_rep_layer.project_end",
        "span_rep.span_rep_layer.out_project",
        "count_embed.transformer.out_projector",
    ]

    for prefix in sequential_prefixes:
        if key.startswith(prefix + "."):
            suffix = key[len(prefix) + 1:]
            match = _re.match(r"^(\d+)\.(.*)", suffix)
            if match:
                idx = match.group(1)
                rest = match.group(2)
                key = f"{prefix}.layers.{idx}.{rest}"
            break

    return key


def convert_gru_weights(torch_weights: Dict[str, np.ndarray], prefix: str) -> Dict[str, np.ndarray]:
    """
    Convert PyTorch GRU weights to MLX GRU format.

    PyTorch GRU stores:
        {prefix}.weight_ih_l0: (3*hidden, input)
        {prefix}.weight_hh_l0: (3*hidden, hidden)
        {prefix}.bias_ih_l0: (3*hidden,)
        {prefix}.bias_hh_l0: (3*hidden,)

    MLX GRU stores (see mlx.nn.GRU source):
        {prefix}.Wx: (3*hidden, input)  -- same layout, no transpose
        {prefix}.Wh: (3*hidden, hidden) -- same layout, no transpose
        {prefix}.b: (3*hidden,)   -- combined bias (ih + hh for reset/update gates)
        {prefix}.bhn: (hidden,)   -- separate bias for new gate hidden state
    """
    result = {}

    wih_key = f"{prefix}.weight_ih_l0"
    whh_key = f"{prefix}.weight_hh_l0"
    bih_key = f"{prefix}.bias_ih_l0"
    bhh_key = f"{prefix}.bias_hh_l0"

    if wih_key in torch_weights:
        result[f"{prefix}.Wx"] = torch_weights[wih_key]
    if whh_key in torch_weights:
        result[f"{prefix}.Wh"] = torch_weights[whh_key]

    # MLX GRU splits biases: b covers reset+update gates from both ih and hh,
    # plus the new gate input bias. bhn is the new gate hidden bias.
    if bih_key in torch_weights and bhh_key in torch_weights:
        bih = torch_weights[bih_key]
        bhh = torch_weights[bhh_key]
        hidden_size = bih.shape[0] // 3

        # Split into gates: [reset, update, new] each of size hidden_size
        bih_r, bih_z, bih_n = np.split(bih, 3)
        bhh_r, bhh_z, bhh_n = np.split(bhh, 3)

        # MLX b = [bih_r + bhh_r, bih_z + bhh_z, bih_n] (3*hidden)
        b = np.concatenate([bih_r + bhh_r, bih_z + bhh_z, bih_n])
        result[f"{prefix}.b"] = b

        # MLX bhn = bhh_n (hidden)
        result[f"{prefix}.bhn"] = bhh_n

    return result


def split_in_proj_weight(
    torch_weights: Dict[str, np.ndarray],
    prefix: str,
    d_model: int,
) -> Dict[str, np.ndarray]:
    """
    Split PyTorch TransformerEncoderLayer's combined in_proj into
    separate q_proj, k_proj, v_proj for our MLX implementation.

    PyTorch stores:
        {prefix}.self_attn.in_proj_weight: (3*d_model, d_model)
        {prefix}.self_attn.in_proj_bias: (3*d_model,)

    We need:
        {prefix}.q_proj.weight: (d_model, d_model)
        {prefix}.k_proj.weight: (d_model, d_model)
        {prefix}.v_proj.weight: (d_model, d_model)
        + biases
    """
    result = {}

    w_key = f"{prefix}.self_attn.in_proj_weight"
    b_key = f"{prefix}.self_attn.in_proj_bias"

    if w_key in torch_weights:
        w = torch_weights[w_key]
        wq, wk, wv = np.split(w, 3, axis=0)
        result[f"{prefix}.q_proj.weight"] = wq
        result[f"{prefix}.k_proj.weight"] = wk
        result[f"{prefix}.v_proj.weight"] = wv

    if b_key in torch_weights:
        b = torch_weights[b_key]
        bq, bk, bv = np.split(b, 3)
        result[f"{prefix}.q_proj.bias"] = bq
        result[f"{prefix}.k_proj.bias"] = bk
        result[f"{prefix}.v_proj.bias"] = bv

    return result


def convert_weights(
    torch_weights: Dict[str, np.ndarray],
    encoder_config: DebertaV2Config,
) -> Dict[str, mx.array]:
    """
    Convert all PyTorch weights to MLX format.

    Handles key remapping, GRU conversion, and in_proj splitting.
    """
    mlx_weights = {}

    # Phase 1: Identify and convert GRU weights
    gru_prefixes = set()
    for key in torch_weights:
        if ".weight_ih_l0" in key:
            prefix = key.replace(".weight_ih_l0", "")
            gru_prefixes.add(prefix)

    gru_keys = set()
    for prefix in gru_prefixes:
        gru_converted = convert_gru_weights(torch_weights, prefix)
        for k, v in gru_converted.items():
            mlx_weights[k] = mx.array(v)
        gru_keys.update([
            f"{prefix}.weight_ih_l0",
            f"{prefix}.weight_hh_l0",
            f"{prefix}.bias_ih_l0",
            f"{prefix}.bias_hh_l0",
        ])

    # Phase 2: Identify and split in_proj weights from TransformerEncoderLayer
    in_proj_prefixes = set()
    for key in torch_weights:
        if ".self_attn.in_proj_weight" in key:
            prefix = key.rsplit(".self_attn.in_proj_weight", 1)[0]
            in_proj_prefixes.add(prefix)

    in_proj_keys = set()
    for prefix in in_proj_prefixes:
        w_key = f"{prefix}.self_attn.in_proj_weight"
        d_model = torch_weights[w_key].shape[1]
        split_result = split_in_proj_weight(torch_weights, prefix, d_model)
        for k, v in split_result.items():
            remapped = remap_key(k)
            mlx_weights[remapped] = mx.array(v)
        in_proj_keys.add(f"{prefix}.self_attn.in_proj_weight")
        in_proj_keys.add(f"{prefix}.self_attn.in_proj_bias")

    # Phase 3: Convert remaining weights with key remapping
    skip_keys = gru_keys | in_proj_keys
    for key, value in torch_weights.items():
        if key in skip_keys:
            continue

        new_key = remap_key(key)
        mlx_weights[new_key] = mx.array(value)

    return mlx_weights


def convert(
    repo_id: str = "fastino/gliner2-base-v1",
    output_dir: str = None,
    download_dir: str = None,
):
    """
    Download and convert a GLiNER2 model to MLX format.

    Args:
        repo_id: HuggingFace model repository ID
        output_dir: Where to save the converted model
        download_dir: Where to download the original model
    """
    if output_dir is None:
        output_dir = os.path.join("mlx_models", repo_id.replace("/", "_"))

    print(f"Downloading {repo_id}...")
    model_dir = download_model(repo_id, download_dir or output_dir)

    # Load configs
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        config = json.load(f)

    encoder_config_path = os.path.join(model_dir, "encoder_config", "config.json")
    with open(encoder_config_path) as f:
        encoder_config_dict = json.load(f)

    encoder_config = DebertaV2Config.from_dict(encoder_config_dict)

    print(f"Loading PyTorch weights...")
    torch_weights = load_torch_weights(model_dir)
    print(f"  Found {len(torch_weights)} weight tensors")

    print(f"Converting weights to MLX format...")
    mlx_weights = convert_weights(torch_weights, encoder_config)
    print(f"  Converted {len(mlx_weights)} weight tensors")

    # Save MLX weights
    weights_path = os.path.join(output_dir, "mlx_weights.safetensors")
    print(f"Saving MLX weights to {weights_path}...")
    mx.save_safetensors(weights_path, mlx_weights)

    # Save MLX-specific config
    mlx_config = {
        "model_type": "gliner2",
        "max_width": config.get("max_width", 8),
        "counting_layer": config.get("counting_layer", "count_lstm_v2"),
        "token_pooling": config.get("token_pooling", "first"),
        "encoder_config": encoder_config_dict,
    }
    mlx_config_path = os.path.join(output_dir, "mlx_config.json")
    with open(mlx_config_path, "w") as f:
        json.dump(mlx_config, f, indent=2)

    print(f"Conversion complete! Model saved to {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(description="Convert GLiNER2 model to MLX format")
    parser.add_argument(
        "--repo-id",
        type=str,
        default="fastino/gliner2-base-v1",
        help="HuggingFace model repo ID",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for converted model",
    )
    args = parser.parse_args()
    convert(repo_id=args.repo_id, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
