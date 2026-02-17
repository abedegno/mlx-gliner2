"""
GLiNER2 inference engine for MLX.

Provides the main GLiNER2 class with extraction methods:
  - extract_entities
  - classify_text
  - extract_json
  - extract_relations
  - extract (multi-task schema)
  - batch_extract_entities / batch_extract_relations
"""

from __future__ import annotations

import json
import math
import os
import re
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Literal,
    Optional,
    Pattern,
    Tuple,
    Union,
)

import mlx.core as mx
import mlx.nn as nn

from .encoder.deberta_v2 import DebertaV2Config
from .model import Extractor
from .processor import PreprocessedBatch, SchemaTransformer


# =============================================================================
# Validators
# =============================================================================


@dataclass
class RegexValidator:
    """Regex-based span filter for post-processing."""

    pattern: str | Pattern[str]
    mode: Literal["full", "partial"] = "full"
    exclude: bool = False
    flags: int = re.IGNORECASE
    _compiled: Pattern[str] = field(init=False, repr=False)

    def __post_init__(self):
        if self.mode not in {"full", "partial"}:
            raise ValueError(f"mode must be 'full' or 'partial', got {self.mode!r}")
        compiled = (
            self.pattern
            if isinstance(self.pattern, re.Pattern)
            else re.compile(self.pattern, self.flags)
        )
        object.__setattr__(self, "_compiled", compiled)

    def __call__(self, text: str) -> bool:
        return self.validate(text)

    def validate(self, text: str) -> bool:
        matcher = self._compiled.fullmatch if self.mode == "full" else self._compiled.search
        matched = matcher(text) is not None
        return not matched if self.exclude else matched


# =============================================================================
# Schema Builder
# =============================================================================


class StructureBuilder:
    """Builder for structured data schemas."""

    def __init__(self, schema: "Schema", parent: str):
        self.schema = schema
        self.parent = parent
        self.fields = OrderedDict()
        self.descriptions = OrderedDict()
        self.field_order = []
        self._finished = False

    def field(
        self,
        name: str,
        dtype: Literal["str", "list"] = "list",
        choices: Optional[List[str]] = None,
        description: Optional[str] = None,
        threshold: Optional[float] = None,
        validators: Optional[List[RegexValidator]] = None,
    ) -> "StructureBuilder":
        self.fields[name] = {"value": "", "choices": choices} if choices else ""
        self.field_order.append(name)

        if description:
            self.descriptions[name] = description

        self.schema._store_field_metadata(
            self.parent, name, dtype, threshold, choices, validators
        )
        return self

    def _auto_finish(self):
        if not self._finished:
            self.schema._store_field_order(self.parent, self.field_order)
            self.schema.schema["json_structures"].append({self.parent: self.fields})

            if self.descriptions:
                if "json_descriptions" not in self.schema.schema:
                    self.schema.schema["json_descriptions"] = {}
                self.schema.schema["json_descriptions"][self.parent] = self.descriptions

            self._finished = True

    def __getattr__(self, name):
        if hasattr(self.schema, name):
            self._auto_finish()
            return getattr(self.schema, name)
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{name}'")


class Schema:
    """Schema builder for extraction tasks."""

    def __init__(self):
        self.schema = {
            "json_structures": [],
            "classifications": [],
            "entities": OrderedDict(),
            "relations": [],
            "json_descriptions": {},
            "entity_descriptions": OrderedDict(),
        }
        self._field_metadata = {}
        self._entity_metadata = {}
        self._relation_metadata = {}
        self._field_orders = {}
        self._entity_order = []
        self._relation_order = []
        self._active_builder = None

    def _store_field_metadata(self, parent, field_name, dtype, threshold, choices, validators=None):
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be 0-1, got {threshold}")
        self._field_metadata[f"{parent}.{field_name}"] = {
            "dtype": dtype,
            "threshold": threshold,
            "choices": choices,
            "validators": validators or [],
        }

    def _store_entity_metadata(self, entity, dtype, threshold):
        if threshold is not None and not 0 <= threshold <= 1:
            raise ValueError(f"Threshold must be 0-1, got {threshold}")
        self._entity_metadata[entity] = {"dtype": dtype, "threshold": threshold}

    def _store_field_order(self, parent, order):
        self._field_orders[parent] = order

    def structure(self, name: str) -> StructureBuilder:
        if self._active_builder:
            self._active_builder._auto_finish()
        self._active_builder = StructureBuilder(self, name)
        return self._active_builder

    def classification(
        self,
        task: str,
        labels: Union[List[str], Dict[str, str]],
        multi_label: bool = False,
        cls_threshold: float = 0.5,
        **kwargs,
    ) -> "Schema":
        if self._active_builder:
            self._active_builder._auto_finish()
            self._active_builder = None

        label_names = list(labels.keys()) if isinstance(labels, dict) else labels
        label_descs = labels if isinstance(labels, dict) else None

        config = {
            "task": task,
            "labels": label_names,
            "multi_label": multi_label,
            "cls_threshold": cls_threshold,
            "true_label": ["N/A"],
            **kwargs,
        }
        if label_descs:
            config["label_descriptions"] = label_descs

        self.schema["classifications"].append(config)
        return self

    def entities(
        self,
        entity_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
        dtype: Literal["str", "list"] = "list",
        threshold: Optional[float] = None,
    ) -> "Schema":
        if self._active_builder:
            self._active_builder._auto_finish()
            self._active_builder = None

        entities = self._parse_entity_input(entity_types)

        for name, config in entities.items():
            self.schema["entities"][name] = ""
            if name not in self._entity_order:
                self._entity_order.append(name)

            self._store_entity_metadata(
                name,
                config.get("dtype", dtype),
                config.get("threshold", threshold),
            )

            if "description" in config:
                self.schema["entity_descriptions"][name] = config["description"]

        return self

    def _parse_entity_input(self, entity_types):
        if isinstance(entity_types, str):
            return {entity_types: {}}
        elif isinstance(entity_types, list):
            return {name: {} for name in entity_types}
        elif isinstance(entity_types, dict):
            result = {}
            for name, config in entity_types.items():
                if isinstance(config, str):
                    result[name] = {"description": config}
                elif isinstance(config, dict):
                    result[name] = config
                else:
                    result[name] = {}
            return result
        raise ValueError("Invalid entity_types format")

    def relations(
        self,
        relation_types: Union[str, List[str], Dict[str, Union[str, Dict]]],
        threshold: Optional[float] = None,
    ) -> "Schema":
        if self._active_builder:
            self._active_builder._auto_finish()
            self._active_builder = None

        if isinstance(relation_types, str):
            relations = {relation_types: {}}
        elif isinstance(relation_types, list):
            relations = {name: {} for name in relation_types}
        elif isinstance(relation_types, dict):
            relations = {}
            for name, config in relation_types.items():
                relations[name] = (
                    {"description": config}
                    if isinstance(config, str)
                    else (config if isinstance(config, dict) else {})
                )
        else:
            raise ValueError("Invalid relation_types format")

        for name, config in relations.items():
            self.schema["relations"].append({name: {"head": "", "tail": ""}})
            if name not in self._relation_order:
                self._relation_order.append(name)
            self._field_orders[name] = ["head", "tail"]

            rel_threshold = config.get("threshold", threshold)
            if rel_threshold is not None and not 0 <= rel_threshold <= 1:
                raise ValueError(f"Threshold must be 0-1, got {rel_threshold}")
            self._relation_metadata[name] = {"threshold": rel_threshold}

        return self

    def build(self) -> Dict[str, Any]:
        if self._active_builder:
            self._active_builder._auto_finish()
            self._active_builder = None
        return self.schema


# =============================================================================
# Main GLiNER2 Class
# =============================================================================


class GLiNER2:
    """
    GLiNER2 Information Extraction Model for MLX.

    Usage:
        >>> extractor = GLiNER2.from_pretrained("mlx_models/fastino_gliner2-base-v1")
        >>> result = extractor.extract_entities(
        ...     "Apple CEO Tim Cook announced iPhone 15.",
        ...     ["company", "person", "product"]
        ... )
    """

    def __init__(self, model: Extractor, processor: SchemaTransformer):
        self.model = model
        self.processor = processor

    @classmethod
    def from_pretrained(cls, model_path: str) -> "GLiNER2":
        """
        Load a converted MLX GLiNER2 model.

        Args:
            model_path: Path to directory containing mlx_config.json,
                        mlx_weights.safetensors, and tokenizer files.
        """
        config_path = os.path.join(model_path, "mlx_config.json")
        with open(config_path) as f:
            config = json.load(f)

        encoder_config = DebertaV2Config.from_dict(config["encoder_config"])

        model = Extractor(
            encoder_config=encoder_config,
            max_width=config.get("max_width", 8),
            counting_layer=config.get("counting_layer", "count_lstm_v2"),
            token_pooling=config.get("token_pooling", "first"),
        )

        weights_path = os.path.join(model_path, "mlx_weights.safetensors")
        weights = mx.load(weights_path)
        model.load_weights(list(weights.items()))

        mx.eval(model.parameters())

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        processor = SchemaTransformer(
            tokenizer=tokenizer,
            token_pooling=config.get("token_pooling", "first"),
        )

        return cls(model=model, processor=processor)

    def create_schema(self) -> Schema:
        """Create a new schema builder."""
        return Schema()

    # =========================================================================
    # High-Level Extraction API
    # =========================================================================

    def extract_entities(
        self,
        text: str,
        entity_types: Union[List[str], Dict[str, str]],
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> Dict[str, Any]:
        """Extract named entities from text."""
        schema = self.create_schema().entities(entity_types)
        results = self._extract_single(
            text, schema, threshold, include_confidence, include_spans
        )
        return results.get("entities", {})

    def classify_text(
        self,
        text: str,
        classifications: Dict[str, Any],
        include_confidence: bool = False,
    ) -> Dict[str, Any]:
        """Classify text into categories."""
        schema = self.create_schema()
        for task_name, config in classifications.items():
            if isinstance(config, list):
                schema = schema.classification(task_name, config)
            elif isinstance(config, dict):
                labels = config.get("labels", config.get("options", []))
                multi_label = config.get("multi_label", False)
                cls_threshold = config.get("cls_threshold", 0.5)
                schema = schema.classification(
                    task_name, labels,
                    multi_label=multi_label, cls_threshold=cls_threshold,
                )

        results = self._extract_single(text, schema, 0.5, include_confidence, False)

        # Remove non-classification keys
        return {k: v for k, v in results.items() if k not in ("entities", "relation_extraction")}

    def extract_json(
        self,
        text: str,
        json_schema: Dict[str, List[str]],
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> Dict[str, Any]:
        """Extract structured JSON data from text."""
        schema = self.create_schema()

        for struct_name, fields in json_schema.items():
            builder = schema.structure(struct_name)
            for field_spec in fields:
                parts = field_spec.split("::")
                fname = parts[0]
                dtype = "list"
                description = None
                choices = None

                for part in parts[1:]:
                    if part in ("str", "list"):
                        dtype = part
                    elif part.startswith("[") and part.endswith("]"):
                        choices = [c.strip() for c in part[1:-1].split("|")]
                    else:
                        description = part

                builder = builder.field(
                    fname, dtype=dtype, choices=choices, description=description
                )

        results = self._extract_single(
            text, schema, threshold, include_confidence, include_spans
        )
        return results

    def extract_relations(
        self,
        text: str,
        relation_types: Union[List[str], Dict[str, str]],
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> Dict[str, Any]:
        """Extract relationships between entities."""
        schema = self.create_schema().relations(relation_types)
        results = self._extract_single(
            text, schema, threshold, include_confidence, include_spans
        )
        return results.get("relation_extraction", {})

    def extract(
        self,
        text: str,
        schema: Schema,
        threshold: float = 0.5,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> Dict[str, Any]:
        """Extract using a multi-task schema."""
        return self._extract_single(
            text, schema, threshold, include_confidence, include_spans
        )

    # =========================================================================
    # Batch Extraction
    # =========================================================================

    def batch_extract_entities(
        self,
        texts: List[str],
        entity_types: Union[List[str], Dict[str, str]],
        threshold: float = 0.5,
        batch_size: int = 8,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract entities from multiple texts."""
        schema = self.create_schema().entities(entity_types)
        results = self._batch_extract(
            texts, schema, threshold, batch_size,
            include_confidence, include_spans,
        )
        return [r.get("entities", {}) for r in results]

    def batch_extract_relations(
        self,
        texts: List[str],
        relation_types: Union[List[str], Dict[str, str]],
        threshold: float = 0.5,
        batch_size: int = 8,
        include_confidence: bool = False,
        include_spans: bool = False,
    ) -> List[Dict[str, Any]]:
        """Extract relations from multiple texts."""
        schema = self.create_schema().relations(relation_types)
        results = self._batch_extract(
            texts, schema, threshold, batch_size,
            include_confidence, include_spans,
        )
        return [r.get("relation_extraction", {}) for r in results]

    # =========================================================================
    # Internal Extraction
    # =========================================================================

    def _extract_single(
        self,
        text: str,
        schema: Schema,
        threshold: float,
        include_confidence: bool,
        include_spans: bool,
    ) -> Dict[str, Any]:
        """Extract from a single text."""
        results = self._batch_extract(
            [text], schema, threshold, 1, include_confidence, include_spans
        )
        return results[0] if results else {}

    def _batch_extract(
        self,
        texts: List[str],
        schema: Schema,
        threshold: float,
        batch_size: int,
        include_confidence: bool,
        include_spans: bool,
    ) -> List[Dict[str, Any]]:
        """Extract from multiple texts in batches."""
        if not texts:
            return []

        schema_dict = schema.build()
        metadata = {
            "field_metadata": schema._field_metadata,
            "entity_metadata": schema._entity_metadata,
            "relation_metadata": getattr(schema, "_relation_metadata", {}),
            "field_orders": schema._field_orders,
            "entity_order": schema._entity_order,
            "relation_order": getattr(schema, "_relation_order", []),
            "classification_tasks": [
                c["task"] for c in schema_dict.get("classifications", [])
            ],
        }

        for cls_config in schema_dict.get("classifications", []):
            cls_config.setdefault("true_label", ["N/A"])

        normalized = []
        for text in texts:
            if not text:
                text = "."
            elif not text.endswith((".", "!", "?")):
                text = text + "."
            normalized.append(text)

        all_results = []

        for start in range(0, len(normalized), batch_size):
            batch_texts = normalized[start: start + batch_size]
            batch_data = [(t, schema_dict) for t in batch_texts]
            batch = self.processor.collate_fn_inference(batch_data)

            batch_results = self._extract_from_batch(
                batch, threshold, metadata, include_confidence, include_spans
            )

            for result in batch_results:
                formatted = self._format_results(
                    result, include_confidence,
                    metadata.get("relation_order", []),
                    metadata.get("classification_tasks", []),
                )
                all_results.append(formatted)

        return all_results

    def _extract_from_batch(
        self,
        batch: PreprocessedBatch,
        threshold: float,
        metadata: Dict,
        include_confidence: bool,
        include_spans: bool,
    ) -> List[Dict[str, Any]]:
        """Extract from a preprocessed batch."""
        token_embeddings = self.model.encode(batch.input_ids, batch.attention_mask)
        mx.eval(token_embeddings)

        all_token_embs, all_schema_embs = self.processor.extract_embeddings_from_batch(
            token_embeddings, batch.input_ids, batch
        )

        results = []
        for i in range(len(batch)):
            try:
                sample_result = self._extract_sample(
                    token_embs=all_token_embs[i],
                    schema_embs=all_schema_embs[i],
                    schema_tokens_list=batch.schema_tokens_list[i],
                    task_types=batch.task_types[i],
                    text_tokens=batch.text_tokens[i],
                    original_text=batch.original_texts[i],
                    schema=batch.original_schemas[i],
                    start_mapping=batch.start_mappings[i],
                    end_mapping=batch.end_mappings[i],
                    threshold=threshold,
                    metadata=metadata,
                    include_confidence=include_confidence,
                    include_spans=include_spans,
                )
                results.append(sample_result)
            except Exception as e:
                print(f"Error extracting sample {i}: {e}")
                results.append({})

        return results

    def _extract_sample(
        self,
        token_embs: mx.array,
        schema_embs: List[List[mx.array]],
        schema_tokens_list: List[List[str]],
        task_types: List[str],
        text_tokens: List[str],
        original_text: str,
        schema: Dict,
        start_mapping: List[int],
        end_mapping: List[int],
        threshold: float,
        metadata: Dict,
        include_confidence: bool,
        include_spans: bool,
    ) -> Dict[str, Any]:
        """Extract from a single sample."""
        results = {}

        has_span_task = any(t != "classifications" for t in task_types)
        span_info = None
        if has_span_task and token_embs.size > 0:
            span_info = self.model.compute_span_rep(token_embs)
            mx.eval(span_info["span_rep"])

        cls_fields = {}
        for struct in schema.get("json_structures", []):
            for parent, fields in struct.items():
                for fname, fval in fields.items():
                    if isinstance(fval, dict) and "choices" in fval:
                        cls_fields[f"{parent}.{fname}"] = fval["choices"]

        text_len = len(self.processor._tokenize_text(original_text))

        for i, (schema_tokens, task_type) in enumerate(
            zip(schema_tokens_list, task_types)
        ):
            if len(schema_tokens) < 4 or not schema_embs[i]:
                continue

            schema_name = schema_tokens[2].split(" [DESCRIPTION] ")[0]
            embs = mx.stack(schema_embs[i])
            mx.eval(embs)

            if task_type == "classifications":
                self._extract_classification_result(
                    results, schema_name, schema, embs, schema_tokens
                )
            else:
                self._extract_span_result(
                    results, schema_name, task_type, embs, span_info,
                    schema_tokens, text_tokens, text_len, original_text,
                    start_mapping, end_mapping, threshold, metadata,
                    cls_fields, include_confidence, include_spans,
                )

        return results

    def _extract_classification_result(
        self,
        results: Dict,
        schema_name: str,
        schema: Dict,
        embs: mx.array,
        schema_tokens: List[str],
    ):
        """Extract classification result."""
        cls_config = next(
            c
            for c in schema["classifications"]
            if schema_tokens[2].startswith(c["task"])
        )

        cls_embeds = embs[1:]
        logits = self.model.classifier(cls_embeds).squeeze(-1)
        mx.eval(logits)

        is_multi = cls_config.get("multi_label", False)

        if is_multi:
            probs = mx.sigmoid(logits)
        else:
            probs = mx.softmax(logits, axis=-1)

        mx.eval(probs)
        labels = cls_config["labels"]
        cls_threshold = cls_config.get("cls_threshold", 0.5)

        if is_multi:
            chosen = [
                (labels[j], probs[j].item())
                for j in range(len(labels))
                if probs[j].item() >= cls_threshold
            ]
            if not chosen:
                best = int(mx.argmax(probs).item())
                chosen = [(labels[best], probs[best].item())]
            results[schema_name] = chosen
        else:
            best = int(mx.argmax(probs).item())
            results[schema_name] = (labels[best], probs[best].item())

    def _extract_span_result(
        self,
        results: Dict,
        schema_name: str,
        task_type: str,
        embs: mx.array,
        span_info: Optional[Dict],
        schema_tokens: List[str],
        text_tokens: List[str],
        text_len: int,
        original_text: str,
        start_mapping: List[int],
        end_mapping: List[int],
        threshold: float,
        metadata: Dict,
        cls_fields: Dict,
        include_confidence: bool,
        include_spans: bool,
    ):
        """Extract span-based results."""
        field_names = []
        for j in range(len(schema_tokens) - 1):
            if schema_tokens[j] in ("[E]", "[C]", "[R]"):
                field_names.append(schema_tokens[j + 1])

        if not field_names:
            results[schema_name] = [] if schema_name == "entities" else {}
            return

        count_logits = self.model.count_pred(embs[0:1])
        mx.eval(count_logits)
        pred_count = int(mx.argmax(count_logits, axis=1).item())

        if pred_count <= 0 or span_info is None:
            if schema_name == "entities":
                results[schema_name] = []
            elif task_type == "relations":
                results[schema_name] = []
            else:
                results[schema_name] = {}
            return

        struct_proj = self.model.count_embed(embs[1:], pred_count)
        mx.eval(struct_proj)

        span_scores = mx.sigmoid(
            mx.einsum("lkd,bpd->bplk", span_info["span_rep"], struct_proj)
        )
        mx.eval(span_scores)

        if schema_name == "entities":
            results[schema_name] = self._extract_entities(
                field_names, span_scores, text_len, text_tokens,
                original_text, start_mapping, end_mapping,
                threshold, metadata, include_confidence, include_spans,
            )
        elif task_type == "relations":
            results[schema_name] = self._extract_relations(
                schema_name, field_names, span_scores, pred_count,
                text_len, text_tokens, original_text, start_mapping, end_mapping,
                threshold, metadata, include_confidence, include_spans,
            )
        else:
            results[schema_name] = self._extract_structures(
                schema_name, field_names, span_scores, pred_count,
                text_len, text_tokens, original_text, start_mapping, end_mapping,
                threshold, metadata, cls_fields, include_confidence, include_spans,
            )

    def _extract_entities(
        self,
        entity_names: List[str],
        span_scores: mx.array,
        text_len: int,
        text_tokens: List[str],
        text: str,
        start_map: List[int],
        end_map: List[int],
        threshold: float,
        metadata: Dict,
        include_confidence: bool,
        include_spans: bool,
    ) -> List[Dict]:
        """Extract entity results."""
        scores_np = span_scores[0, :, -text_len:]
        entity_results = OrderedDict()

        for name in metadata.get("entity_order", entity_names):
            if name not in entity_names:
                continue

            idx = entity_names.index(name)
            meta = metadata.get("entity_metadata", {}).get(name, {})
            meta_threshold = meta.get("threshold")
            ent_threshold = meta_threshold if meta_threshold is not None else threshold
            dtype = meta.get("dtype", "list")

            spans = self._find_spans(
                scores_np[idx], ent_threshold, text_len, text, start_map, end_map
            )

            if dtype == "list":
                entity_results[name] = self._format_spans(
                    spans, include_confidence, include_spans
                )
            else:
                if spans:
                    text_val, conf, char_start, char_end = spans[0]
                    if include_spans and include_confidence:
                        entity_results[name] = {
                            "text": text_val, "confidence": conf,
                            "start": char_start, "end": char_end,
                        }
                    elif include_spans:
                        entity_results[name] = {
                            "text": text_val, "start": char_start, "end": char_end,
                        }
                    elif include_confidence:
                        entity_results[name] = {"text": text_val, "confidence": conf}
                    else:
                        entity_results[name] = text_val
                else:
                    entity_results[name] = (
                        "" if not include_spans and not include_confidence else None
                    )

        return [entity_results] if entity_results else []

    def _extract_relations(
        self,
        rel_name: str,
        field_names: List[str],
        span_scores: mx.array,
        count: int,
        text_len: int,
        text_tokens: List[str],
        text: str,
        start_map: List[int],
        end_map: List[int],
        threshold: float,
        metadata: Dict,
        include_confidence: bool,
        include_spans: bool,
    ) -> List:
        """Extract relation results."""
        instances = []

        rel_threshold = threshold
        if rel_name in metadata.get("relation_metadata", {}):
            meta_threshold = metadata["relation_metadata"][rel_name].get("threshold")
            rel_threshold = meta_threshold if meta_threshold is not None else threshold

        ordered_fields = metadata.get("field_orders", {}).get(rel_name, field_names)

        for inst in range(count):
            scores = span_scores[inst, :, -text_len:]
            values = []
            field_data = []

            for fname in ordered_fields:
                if fname not in field_names:
                    continue
                fidx = field_names.index(fname)
                spans = self._find_spans(
                    scores[fidx], rel_threshold, text_len, text, start_map, end_map
                )

                if spans:
                    text_val, conf, char_start, char_end = spans[0]
                    values.append(text_val)
                    field_data.append({
                        "text": text_val, "confidence": conf,
                        "start": char_start, "end": char_end,
                    })
                else:
                    values.append(None)
                    field_data.append(None)

            if len(values) == 2 and values[0] and values[1]:
                if include_spans and include_confidence:
                    instances.append({"head": field_data[0], "tail": field_data[1]})
                elif include_spans:
                    instances.append({
                        "head": {
                            "text": field_data[0]["text"],
                            "start": field_data[0]["start"],
                            "end": field_data[0]["end"],
                        },
                        "tail": {
                            "text": field_data[1]["text"],
                            "start": field_data[1]["start"],
                            "end": field_data[1]["end"],
                        },
                    })
                elif include_confidence:
                    instances.append({
                        "head": {
                            "text": field_data[0]["text"],
                            "confidence": field_data[0]["confidence"],
                        },
                        "tail": {
                            "text": field_data[1]["text"],
                            "confidence": field_data[1]["confidence"],
                        },
                    })
                else:
                    instances.append((values[0], values[1]))

        return instances

    def _extract_structures(
        self,
        struct_name: str,
        field_names: List[str],
        span_scores: mx.array,
        count: int,
        text_len: int,
        text_tokens: List[str],
        text: str,
        start_map: List[int],
        end_map: List[int],
        threshold: float,
        metadata: Dict,
        cls_fields: Dict,
        include_confidence: bool,
        include_spans: bool,
    ) -> List[Dict]:
        """Extract structured results."""
        instances = []
        ordered_fields = metadata.get("field_orders", {}).get(struct_name, field_names)

        for inst in range(count):
            scores = span_scores[inst, :, -text_len:]
            instance = OrderedDict()

            for fname in ordered_fields:
                if fname not in field_names:
                    continue

                fidx = field_names.index(fname)
                field_key = f"{struct_name}.{fname}"
                meta = metadata.get("field_metadata", {}).get(field_key, {})
                meta_threshold = meta.get("threshold")
                field_threshold = (
                    meta_threshold if meta_threshold is not None else threshold
                )
                dtype = meta.get("dtype", "list")
                validators = meta.get("validators", [])

                if field_key in cls_fields:
                    choices = cls_fields[field_key]
                    prefix_scores = span_scores[inst, fidx, :-text_len]

                    if dtype == "list":
                        selected = []
                        seen = set()
                        for choice in choices:
                            if choice in seen:
                                continue
                            idx = self._find_choice_idx(
                                choice, text_tokens[:-text_len]
                            )
                            if 0 <= idx < prefix_scores.shape[0]:
                                score = prefix_scores[idx, 0].item()
                                if score >= field_threshold:
                                    if include_confidence:
                                        selected.append(
                                            {"text": choice, "confidence": score}
                                        )
                                    else:
                                        selected.append(choice)
                                    seen.add(choice)
                        instance[fname] = selected
                    else:
                        best = None
                        best_score = -1.0
                        for choice in choices:
                            idx = self._find_choice_idx(
                                choice, text_tokens[:-text_len]
                            )
                            if 0 <= idx < prefix_scores.shape[0]:
                                score = prefix_scores[idx, 0].item()
                                if score > best_score:
                                    best_score = score
                                    best = choice
                        if best and best_score >= field_threshold:
                            if include_confidence:
                                instance[fname] = {
                                    "text": best, "confidence": best_score,
                                }
                            else:
                                instance[fname] = best
                        else:
                            instance[fname] = None
                else:
                    spans = self._find_spans(
                        scores[fidx], field_threshold, text_len, text,
                        start_map, end_map,
                    )

                    if validators:
                        spans = [
                            s for s in spans if all(v.validate(s[0]) for v in validators)
                        ]

                    if dtype == "list":
                        instance[fname] = self._format_spans(
                            spans, include_confidence, include_spans
                        )
                    else:
                        if spans:
                            text_val, conf, char_start, char_end = spans[0]
                            if include_spans and include_confidence:
                                instance[fname] = {
                                    "text": text_val, "confidence": conf,
                                    "start": char_start, "end": char_end,
                                }
                            elif include_spans:
                                instance[fname] = {
                                    "text": text_val, "start": char_start,
                                    "end": char_end,
                                }
                            elif include_confidence:
                                instance[fname] = {
                                    "text": text_val, "confidence": conf,
                                }
                            else:
                                instance[fname] = text_val
                        else:
                            instance[fname] = None

            if any(v is not None and v != [] for v in instance.values()):
                instances.append(instance)

        return instances

    # =========================================================================
    # Span Utilities
    # =========================================================================

    def _find_spans(
        self,
        scores: mx.array,
        threshold: float,
        text_len: int,
        text: str,
        start_map: List[int],
        end_map: List[int],
    ) -> List[Tuple[str, float, int, int]]:
        """Find valid spans above threshold."""
        scores_np = scores.tolist()
        spans = []

        for start in range(len(scores_np)):
            if not isinstance(scores_np[start], list):
                continue
            for width in range(len(scores_np[start])):
                score = scores_np[start][width]
                if score < threshold:
                    continue
                end = start + width + 1
                if 0 <= start < text_len and end <= text_len:
                    try:
                        char_start = start_map[start]
                        char_end = end_map[end - 1]
                        text_span = text[char_start:char_end].strip()
                    except (IndexError, KeyError):
                        continue
                    if text_span:
                        spans.append((text_span, score, char_start, char_end))

        return spans

    def _format_spans(
        self,
        spans: List[Tuple],
        include_confidence: bool,
        include_spans: bool = False,
    ) -> List:
        """Format spans with overlap removal."""
        if not spans:
            return []

        sorted_spans = sorted(spans, key=lambda x: x[1], reverse=True)
        selected = []

        for text, conf, start, end in sorted_spans:
            overlap = any(not (end <= s[2] or start >= s[3]) for s in selected)
            if not overlap:
                selected.append((text, conf, start, end))

        if include_spans and include_confidence:
            return [
                {"text": s[0], "confidence": s[1], "start": s[2], "end": s[3]}
                for s in selected
            ]
        elif include_spans:
            return [{"text": s[0], "start": s[2], "end": s[3]} for s in selected]
        elif include_confidence:
            return [{"text": s[0], "confidence": s[1]} for s in selected]
        else:
            return [s[0] for s in selected]

    def _find_choice_idx(self, choice: str, tokens: List[str]) -> int:
        choice_lower = choice.lower()
        for i, tok in enumerate(tokens):
            if tok.lower() == choice_lower or choice_lower in tok.lower():
                return i
        return -1

    # =========================================================================
    # Result Formatting
    # =========================================================================

    def _format_results(
        self,
        results: Dict,
        include_confidence: bool = False,
        requested_relations: List[str] = None,
        classification_tasks: List[str] = None,
    ) -> Dict[str, Any]:
        """Format extraction results into final output."""
        formatted = {}
        relations = {}
        requested_relations = requested_relations or []
        classification_tasks = classification_tasks or []

        for key, value in results.items():
            is_classification = key in classification_tasks
            is_relation = False

            if not is_classification:
                if key in requested_relations:
                    is_relation = True
                elif isinstance(value, list) and len(value) > 0:
                    if isinstance(value[0], tuple) and len(value[0]) == 2:
                        is_relation = True
                    elif (
                        isinstance(value[0], dict)
                        and "head" in value[0]
                        and "tail" in value[0]
                    ):
                        is_relation = True

            if is_classification:
                if isinstance(value, list):
                    if include_confidence:
                        formatted[key] = [
                            {"label": l, "confidence": c} for l, c in value
                        ]
                    else:
                        formatted[key] = [l for l, _ in value]
                elif isinstance(value, tuple):
                    label, conf = value
                    formatted[key] = (
                        {"label": label, "confidence": conf}
                        if include_confidence
                        else label
                    )
                else:
                    formatted[key] = value
            elif is_relation:
                relations[key] = value if isinstance(value, list) else []
            elif isinstance(value, list):
                if len(value) == 0:
                    formatted[key] = {} if key == "entities" else value
                elif isinstance(value[0], dict):
                    if key == "entities":
                        formatted[key] = self._format_entity_dict(
                            value[0], include_confidence
                        )
                    else:
                        formatted[key] = value
                else:
                    formatted[key] = value
            else:
                formatted[key] = value

        # Ensure all requested relations appear
        for rel in requested_relations:
            if rel not in relations:
                relations[rel] = []

        if relations:
            formatted["relation_extraction"] = relations

        return formatted

    def _format_entity_dict(
        self, entity_dict: Dict, include_confidence: bool
    ) -> Dict:
        formatted = {}
        for name, values in entity_dict.items():
            if isinstance(values, list):
                if include_confidence:
                    formatted[name] = values
                else:
                    formatted[name] = [
                        v["text"] if isinstance(v, dict) and "text" in v else v
                        for v in values
                    ]
            else:
                formatted[name] = values
        return formatted
