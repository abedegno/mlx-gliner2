"""
Test entity extraction with include_confidence and include_spans parameters.

Ported from upstream GLiNER2 tests, adapted for the mlx_gliner2 API:
  - extract_entities() returns the entities dict directly (no "entities" wrapper)
  - Model path is read from MLX_GLINER2_MODEL env var

Requires a converted model. Run conversion first:
    python -m mlx_gliner2.convert --repo-id fastino/gliner2-base-v1
"""

import json
import os

import pytest

from mlx_gliner2 import GLiNER2

MODEL_PATH = os.environ.get(
    "MLX_GLINER2_MODEL", "mlx_models/fastino_gliner2-base-v1"
)

requires_model = pytest.mark.skipif(
    not os.path.isdir(MODEL_PATH),
    reason=f"Converted model not found at {MODEL_PATH}",
)


@pytest.fixture(scope="module")
def model():
    return GLiNER2.from_pretrained(MODEL_PATH)


TEXT = "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino on September 12."
ENTITY_TYPES = ["company", "person", "product", "location", "date"]


@requires_model
def test_basic_extraction(model):
    """Basic entity extraction returns a dict mapping entity types to lists."""
    result = model.extract_entities(TEXT, ENTITY_TYPES)

    assert isinstance(result, dict)
    for key in result:
        assert key in ENTITY_TYPES
        assert isinstance(result[key], list)

    print("\nBasic extraction:")
    print(json.dumps(result, indent=2))


@requires_model
def test_with_confidence(model):
    """include_confidence adds confidence scores to each entity."""
    result = model.extract_entities(TEXT, ENTITY_TYPES, include_confidence=True)

    assert isinstance(result, dict)
    for key, entities in result.items():
        assert isinstance(entities, list)
        for ent in entities:
            assert isinstance(ent, dict)
            assert "text" in ent
            assert "confidence" in ent
            assert 0.0 <= ent["confidence"] <= 1.0

    print("\nWith confidence:")
    print(json.dumps(result, indent=2))


@requires_model
def test_with_spans(model):
    """include_spans adds character-level start/end positions."""
    result = model.extract_entities(TEXT, ENTITY_TYPES, include_spans=True)

    assert isinstance(result, dict)
    for key, entities in result.items():
        assert isinstance(entities, list)
        for ent in entities:
            assert isinstance(ent, dict)
            assert "text" in ent
            assert "start" in ent
            assert "end" in ent
            assert isinstance(ent["start"], int)
            assert isinstance(ent["end"], int)

    print("\nWith spans:")
    print(json.dumps(result, indent=2))


@requires_model
def test_with_confidence_and_spans(model):
    """Both confidence and spans together."""
    result = model.extract_entities(
        TEXT, ENTITY_TYPES, include_confidence=True, include_spans=True
    )

    assert isinstance(result, dict)
    for key, entities in result.items():
        for ent in entities:
            assert {"text", "confidence", "start", "end"} <= set(ent.keys())

    print("\nWith confidence and spans:")
    print(json.dumps(result, indent=2))


@requires_model
def test_span_positions_match_text(model):
    """Verify that character positions actually match the extracted text."""
    result = model.extract_entities(
        TEXT, ENTITY_TYPES, include_confidence=True, include_spans=True
    )

    print("\nSpan verification:")
    for entity_type, entities in result.items():
        for ent in entities:
            extracted = TEXT[ent["start"]:ent["end"]]
            print(
                f"  {entity_type}: '{ent['text']}' at "
                f"[{ent['start']}:{ent['end']}] -> '{extracted}'"
            )
            assert extracted == ent["text"], (
                f"Span mismatch for {entity_type}: "
                f"expected '{ent['text']}', got '{extracted}'"
            )


@requires_model
def test_batch_entity_extraction(model):
    """Batch entity extraction returns one result per input text."""
    texts = [
        "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino.",
        "Google's Sundar Pichai spoke at the conference in Mountain View.",
        "Microsoft released Windows 11 in Redmond.",
    ]
    entity_types = ["company", "person", "product", "location"]

    results = model.batch_extract_entities(texts, entity_types, batch_size=2)

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, dict)

    print("\nBatch extraction:")
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"\n  Text {i + 1}: {text}")
        print(f"  Result: {json.dumps(result, indent=2)}")


@requires_model
def test_batch_with_confidence_and_spans(model):
    """Batch extraction with full metadata."""
    texts = [
        "Apple CEO Tim Cook announced the iPhone 15 launch in Cupertino.",
        "Google's Sundar Pichai spoke at the conference in Mountain View.",
        "Microsoft released Windows 11 in Redmond.",
    ]
    entity_types = ["company", "person", "product", "location"]

    results = model.batch_extract_entities(
        texts, entity_types, batch_size=2,
        include_confidence=True, include_spans=True,
    )

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, dict)
        for key, entities in result.items():
            for ent in entities:
                assert isinstance(ent, dict)
                assert {"text", "confidence", "start", "end"} <= set(ent.keys())

    print("\nBatch with confidence and spans:")
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"\n  Text {i + 1}: {text}")
        print(f"  Result: {json.dumps(result, indent=2)}")


if __name__ == "__main__":
    if not os.path.isdir(MODEL_PATH):
        print(f"Model not found at {MODEL_PATH}. Run conversion first.")
        raise SystemExit(1)

    m = GLiNER2.from_pretrained(MODEL_PATH)
    test_basic_extraction(m)
    test_with_confidence(m)
    test_with_spans(m)
    test_with_confidence_and_spans(m)
    test_span_positions_match_text(m)
    test_batch_entity_extraction(m)
    test_batch_with_confidence_and_spans(m)
    print("\nAll entity extraction tests passed!")
