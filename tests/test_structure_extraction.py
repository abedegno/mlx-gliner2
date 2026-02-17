"""
Test structure extraction with include_confidence and include_spans parameters.

Ported from upstream GLiNER2 tests, adapted for the mlx_gliner2 API:
  - extract() and extract_json() return the full results dict
  - No batch_extract() method; batch tests use the schema-based API directly
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


TEXT = (
    "Apple announced a new iPhone 15 Pro Max at $1099 "
    "during their September event in Cupertino."
)
FIELDS = ["company", "product", "price", "date", "location"]


def _make_schema(model):
    """Build a structure schema with list-type fields."""
    schema = model.create_schema()
    builder = schema.structure("product_announcement")
    for f in FIELDS:
        builder = builder.field(f)
    return schema


@requires_model
def test_basic_extraction(model):
    """Basic structure extraction returns field lists."""
    schema = _make_schema(model)
    result = model.extract(TEXT, schema)

    assert isinstance(result, dict)
    if "product_announcement" in result:
        instances = result["product_announcement"]
        assert isinstance(instances, list)
        for inst in instances:
            assert isinstance(inst, dict)
            for key in inst:
                assert key in FIELDS

    print("\nBasic extraction:")
    print(json.dumps(result, indent=2))


@requires_model
def test_with_confidence(model):
    """include_confidence adds scores to extracted values."""
    schema = _make_schema(model)
    result = model.extract(TEXT, schema, include_confidence=True)

    assert isinstance(result, dict)
    if "product_announcement" in result:
        for inst in result["product_announcement"]:
            for fname, fval in inst.items():
                if isinstance(fval, list):
                    for item in fval:
                        assert isinstance(item, dict)
                        assert "text" in item
                        assert "confidence" in item

    print("\nWith confidence:")
    print(json.dumps(result, indent=2))


@requires_model
def test_with_spans(model):
    """include_spans adds character positions to extracted values."""
    schema = _make_schema(model)
    result = model.extract(TEXT, schema, include_spans=True)

    assert isinstance(result, dict)
    if "product_announcement" in result:
        for inst in result["product_announcement"]:
            for fname, fval in inst.items():
                if isinstance(fval, list):
                    for item in fval:
                        assert isinstance(item, dict)
                        assert "text" in item
                        assert "start" in item
                        assert "end" in item

    print("\nWith spans:")
    print(json.dumps(result, indent=2))


@requires_model
def test_with_confidence_and_spans(model):
    """Both confidence and spans together."""
    schema = _make_schema(model)
    result = model.extract(
        TEXT, schema, include_confidence=True, include_spans=True
    )

    assert isinstance(result, dict)
    if "product_announcement" in result:
        for inst in result["product_announcement"]:
            for fname, fval in inst.items():
                if isinstance(fval, list):
                    for item in fval:
                        assert {"text", "confidence", "start", "end"} <= set(
                            item.keys()
                        )

    print("\nWith confidence and spans:")
    print(json.dumps(result, indent=2))


@requires_model
def test_span_positions_match_text(model):
    """Verify that character positions match the extracted text."""
    schema = _make_schema(model)
    result = model.extract(
        TEXT, schema, include_confidence=True, include_spans=True
    )

    print("\nSpan verification:")
    if "product_announcement" in result:
        for inst in result["product_announcement"]:
            for fname, fval in inst.items():
                if not isinstance(fval, list):
                    continue
                for item in fval:
                    extracted = TEXT[item["start"]:item["end"]]
                    print(
                        f"  {fname}: '{item['text']}' at "
                        f"[{item['start']}:{item['end']}] -> '{extracted}'"
                    )
                    assert extracted == item["text"], (
                        f"Span mismatch for {fname}: "
                        f"expected '{item['text']}', got '{extracted}'"
                    )


# =========================================================================
# Single-value fields (dtype="str")
# =========================================================================

STR_TEXT = "Apple announced iPhone 15 at $999 on September 12."
STR_FIELDS = ["company", "product", "price", "date"]


def _make_str_schema(model):
    """Build a structure schema with single-value (str) fields."""
    schema = model.create_schema()
    builder = schema.structure("product_info")
    for f in STR_FIELDS:
        builder = builder.field(f, dtype="str")
    return schema


@requires_model
def test_single_value_basic(model):
    """dtype='str' fields return scalar strings."""
    schema = _make_str_schema(model)
    result = model.extract(STR_TEXT, schema)

    assert isinstance(result, dict)
    if "product_info" in result:
        for inst in result["product_info"]:
            for fname, fval in inst.items():
                assert fval is None or isinstance(fval, str), (
                    f"Expected str or None for {fname}, got {type(fval)}"
                )

    print("\nSingle-value basic:")
    print(json.dumps(result, indent=2))


@requires_model
def test_single_value_with_spans(model):
    """dtype='str' fields with include_spans return dicts."""
    schema = _make_str_schema(model)
    result = model.extract(STR_TEXT, schema, include_spans=True)

    assert isinstance(result, dict)
    if "product_info" in result:
        for inst in result["product_info"]:
            for fname, fval in inst.items():
                if fval is not None:
                    assert isinstance(fval, dict)
                    assert "text" in fval
                    assert "start" in fval
                    assert "end" in fval

    print("\nSingle-value with spans:")
    print(json.dumps(result, indent=2))


@requires_model
def test_single_value_with_confidence_and_spans(model):
    """dtype='str' fields with both flags."""
    schema = _make_str_schema(model)
    result = model.extract(
        STR_TEXT, schema, include_confidence=True, include_spans=True
    )

    assert isinstance(result, dict)
    if "product_info" in result:
        for inst in result["product_info"]:
            for fname, fval in inst.items():
                if fval is not None:
                    assert isinstance(fval, dict)
                    assert {"text", "confidence", "start", "end"} <= set(
                        fval.keys()
                    )

    print("\nSingle-value with confidence and spans:")
    print(json.dumps(result, indent=2))


# =========================================================================
# extract_json() convenience method
# =========================================================================


@requires_model
def test_extract_json(model):
    """extract_json() shorthand produces the same kind of results."""
    result = model.extract_json(
        STR_TEXT,
        {
            "product": [
                "name::str::Full product name",
                "price::str::Retail price",
                "date::str::Announcement date",
            ]
        },
    )

    assert isinstance(result, dict)
    if "product" in result:
        assert isinstance(result["product"], list)

    print("\nextract_json():")
    print(json.dumps(result, indent=2))


@requires_model
def test_extract_json_with_confidence_and_spans(model):
    """extract_json() with full metadata."""
    result = model.extract_json(
        STR_TEXT,
        {
            "product": [
                "name::str::Full product name",
                "price::str::Retail price",
            ]
        },
        include_confidence=True,
        include_spans=True,
    )

    assert isinstance(result, dict)
    if "product" in result:
        for inst in result["product"]:
            for fname, fval in inst.items():
                if fval is not None and isinstance(fval, dict):
                    assert "text" in fval

    print("\nextract_json() with confidence and spans:")
    print(json.dumps(result, indent=2))


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
    test_single_value_basic(m)
    test_single_value_with_spans(m)
    test_single_value_with_confidence_and_spans(m)
    test_extract_json(m)
    test_extract_json_with_confidence_and_spans(m)
    print("\nAll structure extraction tests passed!")
