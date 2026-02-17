"""
MLX GLiNER2 - Information extraction on Apple Silicon.

Port of GLiNER2 (https://github.com/fastino-ai/GLiNER2) to Apple's MLX
framework for efficient CPU/GPU inference on Apple Silicon.

Usage:
    # First, convert the model (one-time):
    from mlx_gliner2.convert import convert
    convert("fastino/gliner2-base-v1")

    # Then use for inference:
    from mlx_gliner2 import GLiNER2

    extractor = GLiNER2.from_pretrained("mlx_models/fastino_gliner2-base-v1")

    # Entity extraction
    result = extractor.extract_entities(
        "Apple CEO Tim Cook announced iPhone 15 in Cupertino.",
        ["company", "person", "product", "location"]
    )

    # Text classification
    result = extractor.classify_text(
        "This movie is fantastic!",
        {"sentiment": ["positive", "negative", "neutral"]}
    )

    # Structured extraction
    result = extractor.extract_json(
        "iPhone 15 Pro for $999 with A17 Pro chip.",
        {"product": ["name::str", "price::str", "features"]}
    )

    # Relation extraction
    result = extractor.extract_relations(
        "John works for Apple Inc.",
        ["works_for", "located_in"]
    )
"""

__version__ = "0.1.0"

from .inference import GLiNER2, RegexValidator, Schema
from .convert import convert
