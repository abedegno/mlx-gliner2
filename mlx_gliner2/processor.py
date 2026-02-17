"""
GLiNER2 Schema Transformer processor ported to MLX.

Handles all preprocessing: text tokenization, schema transformation,
batch collation. Uses mlx.core arrays instead of torch tensors.
"""

import copy
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple

import mlx.core as mx
from transformers import AutoTokenizer


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class TransformedRecord:
    """Single transformed record ready for batching."""

    input_ids: List[int]
    mapped_indices: List[Tuple[str, int, int]]
    schema_tokens_list: List[List[str]]
    text_tokens: List[str]
    structure_labels: List[Any]
    task_types: List[str]
    start_token_idx: List[int]
    end_token_idx: List[int]
    text: str
    schema: Dict[str, Any]
    num_schemas: int = field(init=False)

    def __post_init__(self):
        self.num_schemas = len(self.schema_tokens_list)


@dataclass
class PreprocessedBatch:
    """Batch ready for model inference."""

    input_ids: mx.array
    attention_mask: mx.array
    mapped_indices: List[List[Tuple]]
    schema_counts: List[int]
    original_lengths: List[int]
    structure_labels: List[List[Any]]
    task_types: List[List[str]]
    text_tokens: List[List[str]]
    schema_tokens_list: List[List[List[str]]]
    start_mappings: List[List[int]]
    end_mappings: List[List[int]]
    original_texts: List[str]
    original_schemas: List[Dict]

    def __len__(self) -> int:
        return self.input_ids.shape[0]


# =============================================================================
# Tokenizer
# =============================================================================


class WhitespaceTokenSplitter:
    """Fast regex-based tokenizer for text splitting."""

    _PATTERN = re.compile(
        r"""(?:https?://[^\s]+|www\.[^\s]+)
        |[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}
        |@[a-z0-9_]+
        |\w+(?:[-_]\w+)*
        |\S""",
        re.VERBOSE | re.IGNORECASE,
    )

    def __call__(self, text: str, lower: bool = True) -> Iterator[Tuple[str, int, int]]:
        if lower:
            text = text.lower()
        for m in self._PATTERN.finditer(text):
            yield m.group(), m.start(), m.end()


# =============================================================================
# Main Processor Class
# =============================================================================


class SchemaTransformer:
    """
    Schema-based text transformer for GLiNER2.

    Handles preprocessing for inference: tokenization, schema formatting,
    and batch collation.
    """

    SEP_STRUCT = "[SEP_STRUCT]"
    SEP_TEXT = "[SEP_TEXT]"
    P_TOKEN = "[P]"
    C_TOKEN = "[C]"
    E_TOKEN = "[E]"
    R_TOKEN = "[R]"
    L_TOKEN = "[L]"
    EXAMPLE_TOKEN = "[EXAMPLE]"
    OUTPUT_TOKEN = "[OUTPUT]"
    DESC_TOKEN = "[DESCRIPTION]"

    SPECIAL_TOKENS = [
        SEP_STRUCT, SEP_TEXT, P_TOKEN, C_TOKEN, E_TOKEN,
        R_TOKEN, L_TOKEN, EXAMPLE_TOKEN, OUTPUT_TOKEN, DESC_TOKEN,
    ]

    def __init__(
        self,
        model_name: str = None,
        tokenizer=None,
        token_pooling: str = "first",
    ):
        if model_name is None and tokenizer is None:
            raise ValueError("Either model_name or tokenizer must be provided.")

        self.token_pooling = token_pooling if token_pooling in ["first", "mean", "max"] else "first"
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.word_splitter = WhitespaceTokenSplitter()

        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": self.SPECIAL_TOKENS}
        )

    # =========================================================================
    # Public API
    # =========================================================================

    def collate_fn_inference(
        self, batch: List[Tuple[str, Any]]
    ) -> PreprocessedBatch:
        """Collate function for inference."""
        return self._collate_batch(batch)

    def transform_and_format(
        self, text: str, schema: Dict[str, Any]
    ) -> TransformedRecord:
        """Transform a single record for inference."""
        record = {"text": text, "schema": schema}
        return self._transform_record(record)

    # =========================================================================
    # Internal: Batch Processing
    # =========================================================================

    def _collate_batch(
        self, batch: List[Tuple[str, Any]]
    ) -> PreprocessedBatch:
        """Internal collate implementation."""
        transformed_records = []

        for text, schema in batch:
            if hasattr(schema, "build"):
                schema = schema.build()
            elif hasattr(schema, "schema"):
                schema = schema.schema

            if text and not text.endswith((".", "!", "?")):
                text = text + "."
            elif not text:
                text = "."

            record = {"text": text, "schema": copy.deepcopy(schema)}

            try:
                transformed = self._transform_record(record)
                transformed_records.append(transformed)
            except Exception:
                transformed_records.append(self._create_fallback_record(text, schema))

        return self._pad_batch(transformed_records)

    def _transform_record(self, record: Dict[str, Any]) -> TransformedRecord:
        """Transform a single record."""
        record_ = copy.deepcopy(record)
        text, schema = record_["text"], record_["schema"]

        prefix = self._build_classification_prefix(schema)
        original_schema = copy.deepcopy(schema)

        if prefix:
            self._wrap_classification_fields(schema, prefix)

        text_tokens = []
        start_idx_map = []
        end_idx_map = []
        for tkn, start, end in self.word_splitter(text, lower=True):
            text_tokens.append(tkn)
            start_idx_map.append(start)
            end_idx_map.append(end)

        len_prefix = 0
        if prefix:
            text_tokens = prefix + text_tokens
            len_prefix = len(prefix)

        processed = self._infer_from_json(schema)

        results = self._build_outputs(processed, schema, text_tokens, len_prefix)

        schema_tokens_list = [r["schema_tokens"] for r in results]
        format_result = self._format_input_with_mapping(schema_tokens_list, text_tokens)

        return TransformedRecord(
            input_ids=format_result["input_ids"],
            mapped_indices=format_result["mapped_indices"],
            schema_tokens_list=schema_tokens_list,
            text_tokens=text_tokens,
            structure_labels=[r["output"] for r in results],
            task_types=[r["task_type"] for r in results],
            start_token_idx=start_idx_map,
            end_token_idx=end_idx_map,
            text=text,
            schema=original_schema,
        )

    def _pad_batch(self, records: List[TransformedRecord]) -> PreprocessedBatch:
        """Pad transformed records into a batch."""
        if not records:
            return self._empty_batch()

        max_len = max(len(r.input_ids) for r in records)
        batch_size = len(records)

        input_ids_list = []
        attention_mask_list = []
        original_lengths = []

        for rec in records:
            seq_len = len(rec.input_ids)
            padded_ids = rec.input_ids + [0] * (max_len - seq_len)
            padded_mask = [1] * seq_len + [0] * (max_len - seq_len)
            input_ids_list.append(padded_ids)
            attention_mask_list.append(padded_mask)
            original_lengths.append(seq_len)

        input_ids = mx.array(input_ids_list, dtype=mx.int32)
        attention_mask = mx.array(attention_mask_list, dtype=mx.int32)

        return PreprocessedBatch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            mapped_indices=[r.mapped_indices for r in records],
            schema_counts=[r.num_schemas for r in records],
            original_lengths=original_lengths,
            structure_labels=[r.structure_labels for r in records],
            task_types=[r.task_types for r in records],
            text_tokens=[r.text_tokens for r in records],
            schema_tokens_list=[r.schema_tokens_list for r in records],
            start_mappings=[r.start_token_idx for r in records],
            end_mappings=[r.end_token_idx for r in records],
            original_texts=[r.text for r in records],
            original_schemas=[r.schema for r in records],
        )

    def _empty_batch(self) -> PreprocessedBatch:
        return PreprocessedBatch(
            input_ids=mx.zeros((0, 0), dtype=mx.int32),
            attention_mask=mx.zeros((0, 0), dtype=mx.int32),
            mapped_indices=[],
            schema_counts=[],
            original_lengths=[],
            structure_labels=[],
            task_types=[],
            text_tokens=[],
            schema_tokens_list=[],
            start_mappings=[],
            end_mappings=[],
            original_texts=[],
            original_schemas=[],
        )

    def _create_fallback_record(self, text: str, schema: Dict) -> TransformedRecord:
        dummy_tokens = ["(", "[P]", "dummy", "(", "[E]", "entity", ")", ")"]
        format_result = self._format_input_with_mapping([dummy_tokens], ["."])

        return TransformedRecord(
            input_ids=format_result["input_ids"],
            mapped_indices=format_result["mapped_indices"],
            schema_tokens_list=[dummy_tokens],
            text_tokens=["."],
            structure_labels=[[1, [[(0, 0)]]]],
            task_types=["entities"],
            start_token_idx=[0],
            end_token_idx=[1],
            text=text or ".",
            schema=schema or {},
        )

    # =========================================================================
    # Internal: Schema Processing
    # =========================================================================

    def _build_classification_prefix(self, schema: Dict[str, Any]) -> List[str]:
        prefix_tokens = []

        for struct in schema.get("json_structures", []):
            for parent, fields in struct.items():
                cls_fields = [
                    (fname, fval)
                    for fname, fval in fields.items()
                    if isinstance(fval, dict) and "value" in fval and "choices" in fval
                ]

                inner = []
                for fname, fval in cls_fields:
                    choices = fval["choices"].copy()
                    choice_tokens = []
                    for i, c in enumerate(choices):
                        if i > 0:
                            choice_tokens.append("|")
                        choice_tokens.append(c)
                    inner.extend([fname, "("] + choice_tokens + [")", ","])

                if inner:
                    inner = inner[:-1]
                    prefix_tokens.extend(["(", f"{parent}:", *inner, ")"])

        return prefix_tokens

    def _wrap_classification_fields(self, schema: Dict, prefix: List[str]):
        def wrap(val):
            if isinstance(val, list):
                return [f"[selection]{v}" for v in val]
            return f"[selection]{val}"

        cls_keys = {
            f"{parent}.{fname}"
            for struct in schema.get("json_structures", [])
            for parent, fields in struct.items()
            for fname, fval in fields.items()
            if isinstance(fval, dict) and {"value", "choices"} <= fval.keys()
        }

        for struct in schema.get("json_structures", []):
            for parent, fields in struct.items():
                for fname in list(fields):
                    key = f"{parent}.{fname}"
                    if key not in cls_keys:
                        continue
                    fval = fields[fname]
                    raw = fval["value"] if isinstance(fval, dict) else fval
                    fields[fname] = wrap(raw)

    def _infer_from_json(self, schema: Dict[str, Any]) -> Dict[str, Any]:
        schemas = []
        labels = []
        types = []

        self._process_json_structures(schema, schemas, labels, types)
        self._process_entities(schema, schemas, labels, types)
        self._process_relations(schema, schemas, labels, types)
        self._process_classifications(schema, schemas, labels, types)

        return {
            "schemas": schemas,
            "structure_labels": labels,
            "task_types": types,
            "new_schema": schema,
        }

    def _process_json_structures(self, schema, schemas, labels, types):
        if "json_structures" not in schema:
            return

        json_descs = schema.get("json_descriptions", {})
        groups = {}

        for item in schema["json_structures"]:
            for parent, fields in item.items():
                groups.setdefault(parent, []).append(fields)

        for parent, occurrences in groups.items():
            all_fields = set()
            for occ in occurrences:
                all_fields.update(occ.keys())
            chosen = list(all_fields)

            if not chosen:
                continue

            spans = []
            for occ in occurrences:
                span = [occ.get(f) for f in chosen]
                spans.append(span)

            uniq = []
            seen = set()
            for s in spans:
                key = tuple(tuple(x) if isinstance(x, list) else x for x in s)
                if key not in seen:
                    uniq.append(s)
                    seen.add(key)

            if all(all(c is None or c == "" for c in span) for span in uniq):
                count = 0
                uniq = []
            else:
                count = len(uniq)

            labels.append([count, uniq])

            descs = json_descs.get(parent, {})
            mode = "descriptions" if descs else "none"

            schemas.append(
                self._transform_schema(
                    parent, chosen, self.C_TOKEN,
                    label_descriptions=descs, example_mode=mode,
                )
            )
            types.append("json_structures")

    def _process_entities(self, schema, schemas, labels, types):
        if "entities" not in schema:
            return

        entity_fields = list(schema["entities"].keys())
        descs = schema.get("entity_descriptions", {})

        if entity_fields:
            span = [schema["entities"][e] for e in entity_fields]
            labels.append([1, [span]])

            mode = "descriptions" if descs else "none"
            schemas.append(
                self._transform_schema(
                    "entities", entity_fields, self.E_TOKEN,
                    label_descriptions=descs, example_mode=mode,
                )
            )
            types.append("entities")

    def _process_relations(self, schema, schemas, labels, types):
        if "relations" not in schema:
            return

        groups = {}
        for item in schema["relations"]:
            for parent, fields in item.items():
                groups.setdefault(parent, []).append(fields)

        for parent, occurrences in groups.items():
            field_names = list(occurrences[0].keys())

            spans = []
            for occ in occurrences:
                if all(f in occ for f in field_names):
                    spans.append([occ[f] for f in field_names])

            if not spans:
                continue

            seen = set()
            uniq = []
            for span in spans:
                t = tuple(tuple(s) if isinstance(s, list) else s for s in span)
                if t not in seen:
                    seen.add(t)
                    uniq.append(span)

            labels.append([len(uniq), uniq])
            schemas.append(self._transform_schema(parent, field_names, self.R_TOKEN))
            types.append("relations")

    def _process_classifications(self, schema, schemas, labels, types):
        if "classifications" not in schema:
            return

        for idx, item in enumerate(schema["classifications"]):
            cls_labels = item["labels"].copy()
            descs = item.get("label_descriptions", {}) or {}
            examples = item.get("examples", [])

            mode = "both" if examples and descs else ("descriptions" if descs else "none")

            schemas.append(
                self._transform_schema(
                    item["task"], cls_labels, self.L_TOKEN,
                    prompt=item.get("prompt"), examples=examples,
                    label_descriptions=descs, example_mode=mode,
                )
            )
            types.append("classifications")

            true_label = schema["classifications"][idx].get("true_label", [])
            if not isinstance(true_label, list):
                true_label = [true_label]
            schema["classifications"][idx]["true_label"] = true_label
            labels.append([])

    def _transform_schema(
        self,
        parent: str,
        fields: List[str],
        child_prefix: str,
        prompt: str = None,
        examples: List[Tuple[str, str]] = None,
        label_descriptions: Dict[str, str] = None,
        example_mode: str = "both",
    ) -> List[str]:
        prompt_str = parent
        if prompt:
            prompt_str = f"{parent}: {prompt}"

        if example_mode in ["descriptions", "both"] and label_descriptions:
            descs = [(l, d) for l, d in label_descriptions.items() if l in fields]
            for label, desc in descs:
                prompt_str += f" {self.DESC_TOKEN} {label}: {desc}"

        if example_mode in ["few_shot", "both"] and examples:
            for inp, out in examples:
                if out in fields:
                    out_str = out if isinstance(out, str) else ", ".join(out)
                    prompt_str += f" {self.EXAMPLE_TOKEN} {inp} {self.OUTPUT_TOKEN} {out_str}"

        tokens = ["(", self.P_TOKEN, prompt_str, "("]
        for f in fields:
            tokens.extend([child_prefix, f])
        tokens.extend([")", ")"])
        return tokens

    def _build_outputs(
        self,
        processed: Dict,
        schema: Dict,
        text_tokens: List[str],
        len_prefix: int,
    ) -> List[Dict]:
        results = []

        for schema_tokens, task_type, struct_label in zip(
            processed["schemas"],
            processed["task_types"],
            processed["structure_labels"],
        ):
            if task_type != "classifications":
                count, spans = struct_label
                transformed = []

                for span in spans:
                    positions = []
                    for field_val in span:
                        if isinstance(field_val, list):
                            nested = []
                            for sub in field_val:
                                if str(sub).startswith("[selection]"):
                                    pos = self._find_sublist(
                                        [str(sub)[11:]], text_tokens[:len_prefix],
                                        case_insensitive=True,
                                    )
                                else:
                                    pos = self._find_sublist(
                                        self._tokenize_text(str(sub)), text_tokens
                                    )
                                nested.extend(pos)
                            positions.append(nested)
                        else:
                            if str(field_val).startswith("[selection]"):
                                pos = self._find_sublist(
                                    [str(field_val)[11:]], text_tokens[:len_prefix],
                                    case_insensitive=True,
                                )
                            else:
                                pos = self._find_sublist(
                                    self._tokenize_text(str(field_val)), text_tokens
                                )
                            positions.append(pos)
                    transformed.append(positions)

                results.append({
                    "task_type": task_type,
                    "schema_tokens": schema_tokens,
                    "output": [count, transformed],
                })
            else:
                cls_item = next(
                    (c for c in schema["classifications"]
                     if schema_tokens[2].startswith(c["task"])),
                    None,
                )
                if cls_item is None:
                    raise ValueError(f"Missing classification for: {schema_tokens[2]}")

                bool_labels = [
                    1 if l in cls_item["true_label"] else 0
                    for l in cls_item["labels"]
                ]
                results.append({
                    "task_type": task_type,
                    "schema_tokens": schema_tokens,
                    "output": bool_labels,
                })

        return results

    def _find_sublist(
        self,
        sub: List[str],
        lst: List[str],
        case_insensitive: bool = False,
    ) -> List[Tuple[int, int]]:
        if not sub or all(t == "" for t in sub):
            return [(-1, -1)]

        sub_len = len(sub)

        if case_insensitive:
            sub_lower = [s.lower() for s in sub]
            matches = [
                (i, i + sub_len - 1)
                for i in range(len(lst) - sub_len + 1)
                if [t.lower() for t in lst[i : i + sub_len]] == sub_lower
            ]
        else:
            matches = [
                (i, i + sub_len - 1)
                for i in range(len(lst) - sub_len + 1)
                if lst[i : i + sub_len] == sub
            ]
        return matches or [(-1, -1)]

    def _tokenize_text(self, text: str) -> List[str]:
        return [tok for tok, _, _ in self.word_splitter(text, lower=True)]

    # =========================================================================
    # Input Formatting
    # =========================================================================

    def _format_input_with_mapping(
        self,
        schema_tokens_list: List[List[str]],
        text_tokens: List[str],
    ) -> Dict[str, Any]:
        combined = []
        for struct in schema_tokens_list:
            combined.extend(struct)
            combined.append(self.SEP_STRUCT)
        if combined:
            combined.pop()
        combined.append(self.SEP_TEXT)
        combined.extend(text_tokens)

        subwords = []
        mappings = []

        num_schemas = len(schema_tokens_list)
        text_schema_idx = num_schemas
        current_schema = 0
        found_sep = False

        for orig_idx, token in enumerate(combined):
            if token == self.SEP_TEXT:
                seg_type = "sep"
                schema_idx = text_schema_idx
                found_sep = True
            elif not found_sep:
                seg_type = "schema"
                schema_idx = current_schema
                if token == self.SEP_STRUCT:
                    current_schema += 1
            else:
                seg_type = "text"
                schema_idx = text_schema_idx

            sub_tokens = self.tokenizer.tokenize(token)
            subwords.extend(sub_tokens)
            mappings.extend([(seg_type, orig_idx, schema_idx)] * len(sub_tokens))

        input_ids = self.tokenizer.convert_tokens_to_ids(subwords)

        return {
            "input_ids": input_ids,
            "mapped_indices": mappings,
            "subword_list": subwords,
        }

    # =========================================================================
    # Embedding Extraction (Called by Model)
    # =========================================================================

    def extract_embeddings_from_batch(
        self,
        token_embeddings: mx.array,
        input_ids: mx.array,
        batch: PreprocessedBatch,
    ) -> Tuple[List[mx.array], List[List[mx.array]]]:
        """
        Extract token and schema embeddings from encoded batch.

        Args:
            token_embeddings: (batch, seq_len, hidden) from encoder
            input_ids: (batch, seq_len)
            batch: PreprocessedBatch with metadata

        Returns:
            - all_token_embs: List of (text_len, hidden) per sample
            - all_schema_embs: List of schema embeddings per sample
        """
        all_token_embs = []
        all_schema_embs = []

        special_set = {self.P_TOKEN, self.C_TOKEN, self.E_TOKEN, self.R_TOKEN, self.L_TOKEN}

        for i in range(len(batch)):
            seq_len = batch.original_lengths[i]
            embs = token_embeddings[i, :seq_len, :]
            ids = input_ids[i, :seq_len].tolist()
            mappings = batch.mapped_indices[i][:seq_len]
            num_schemas = batch.schema_counts[i]

            schema_embs = [[] for _ in range(num_schemas)]
            word_embs = []
            bucket = []
            last_orig = None

            for j, tid in enumerate(ids):
                seg_type, orig_idx, schema_idx = mappings[j]
                emb = embs[j]

                if seg_type == "schema":
                    tok = self.tokenizer.convert_ids_to_tokens(tid)
                    if tok in special_set:
                        schema_embs[schema_idx].append(emb)

                elif seg_type == "text":
                    if last_orig is not None and orig_idx != last_orig and bucket:
                        word_embs.append(self._aggregate(bucket))
                        bucket = []
                    bucket.append(emb)
                    last_orig = orig_idx

            if bucket:
                word_embs.append(self._aggregate(bucket))

            if word_embs:
                all_token_embs.append(mx.stack(word_embs))
            else:
                all_token_embs.append(
                    mx.zeros((0, embs.shape[-1]))
                )
            all_schema_embs.append(schema_embs)

        return all_token_embs, all_schema_embs

    def _aggregate(self, pieces: List[mx.array]) -> mx.array:
        if self.token_pooling == "first":
            return pieces[0]
        stack = mx.stack(pieces)
        if self.token_pooling == "mean":
            return mx.mean(stack, axis=0)
        if self.token_pooling == "max":
            return mx.max(stack, axis=0)
        return pieces[0]
