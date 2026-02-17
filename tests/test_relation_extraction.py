"""
Test relation extraction with include_confidence and include_spans parameters.

Ported from upstream GLiNER2 tests, adapted for the mlx_gliner2 API:
  - extract_relations() returns the relations dict directly (no wrapper)
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
RELATION_TYPES = ["CEO_of", "located_in", "announced_on"]


@requires_model
def test_basic_extraction(model):
    """Basic relation extraction returns tuples by default."""
    result = model.extract_relations(TEXT, RELATION_TYPES)

    assert isinstance(result, dict)
    for rel_type in RELATION_TYPES:
        if rel_type in result:
            assert isinstance(result[rel_type], list)
            for rel in result[rel_type]:
                assert isinstance(rel, (tuple, list))
                assert len(rel) == 2

    print("\nBasic extraction:")
    print(json.dumps(result, indent=2, default=str))


@requires_model
def test_with_confidence(model):
    """include_confidence returns head/tail dicts with confidence scores."""
    result = model.extract_relations(
        TEXT, RELATION_TYPES, include_confidence=True
    )

    assert isinstance(result, dict)
    for rel_type, relations in result.items():
        for rel in relations:
            assert isinstance(rel, dict)
            assert "head" in rel and "tail" in rel
            assert "text" in rel["head"] and "confidence" in rel["head"]
            assert "text" in rel["tail"] and "confidence" in rel["tail"]
            assert 0.0 <= rel["head"]["confidence"] <= 1.0
            assert 0.0 <= rel["tail"]["confidence"] <= 1.0

    print("\nWith confidence:")
    print(json.dumps(result, indent=2))


@requires_model
def test_with_spans(model):
    """include_spans returns head/tail dicts with character positions."""
    result = model.extract_relations(
        TEXT, RELATION_TYPES, include_spans=True
    )

    assert isinstance(result, dict)
    for rel_type, relations in result.items():
        for rel in relations:
            assert isinstance(rel, dict)
            for role in ("head", "tail"):
                assert "text" in rel[role]
                assert "start" in rel[role]
                assert "end" in rel[role]
                assert isinstance(rel[role]["start"], int)
                assert isinstance(rel[role]["end"], int)

    print("\nWith spans:")
    print(json.dumps(result, indent=2))


@requires_model
def test_with_confidence_and_spans(model):
    """Both confidence and spans together."""
    result = model.extract_relations(
        TEXT, RELATION_TYPES, include_confidence=True, include_spans=True
    )

    assert isinstance(result, dict)
    for rel_type, relations in result.items():
        for rel in relations:
            for role in ("head", "tail"):
                assert {"text", "confidence", "start", "end"} <= set(
                    rel[role].keys()
                )

    print("\nWith confidence and spans:")
    print(json.dumps(result, indent=2))


@requires_model
def test_span_positions_match_text(model):
    """Verify that character positions match the extracted text."""
    result = model.extract_relations(
        TEXT, RELATION_TYPES, include_confidence=True, include_spans=True
    )

    print("\nSpan verification:")
    for rel_type, relations in result.items():
        print(f"\n  {rel_type}:")
        for rel in relations:
            for role in ("head", "tail"):
                part = rel[role]
                extracted = TEXT[part["start"]:part["end"]]
                print(
                    f"    {role}: '{part['text']}' at "
                    f"[{part['start']}:{part['end']}] -> '{extracted}'"
                )
                assert extracted == part["text"], (
                    f"Span mismatch for {rel_type} {role}: "
                    f"expected '{part['text']}', got '{extracted}'"
                )


@requires_model
def test_batch_relation_extraction(model):
    """Batch relation extraction returns one result per input text."""
    texts = [
        "Apple CEO Tim Cook works in Cupertino.",
        "Google CEO Sundar Pichai leads the company in Mountain View.",
        "Microsoft was founded by Bill Gates.",
    ]
    relation_types = ["CEO_of", "works_in", "founded_by"]

    results = model.batch_extract_relations(
        texts, relation_types, batch_size=2
    )

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, dict)

    print("\nBatch extraction:")
    for i, (text, result) in enumerate(zip(texts, results)):
        print(f"\n  Text {i + 1}: {text}")
        print(f"  Result: {json.dumps(result, indent=2, default=str)}")


@requires_model
def test_batch_with_confidence_and_spans(model):
    """Batch relation extraction with full metadata."""
    texts = [
        "Apple CEO Tim Cook works in Cupertino.",
        "Google CEO Sundar Pichai leads the company in Mountain View.",
        "Microsoft was founded by Bill Gates.",
    ]
    relation_types = ["CEO_of", "works_in", "founded_by"]

    results = model.batch_extract_relations(
        texts, relation_types, batch_size=2,
        include_confidence=True, include_spans=True,
    )

    assert len(results) == len(texts)
    for result in results:
        assert isinstance(result, dict)
        for rel_type, relations in result.items():
            for rel in relations:
                for role in ("head", "tail"):
                    assert {"text", "confidence", "start", "end"} <= set(
                        rel[role].keys()
                    )

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
    test_batch_relation_extraction(m)
    test_batch_with_confidence_and_spans(m)
    print("\nAll relation extraction tests passed!")
