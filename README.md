# MLX GLiNER2

Port of [GLiNER2](https://github.com/fastino-ai/GLiNER2) to Apple MLX
for efficient information extraction on Apple Silicon.

## Installation

```bash
pip install mlx-gliner2
```

## Quick Start

```bash
# One-time model conversion
python -m mlx_gliner2.convert --repo-id fastino/gliner2-base-v1
```

```python
from mlx_gliner2 import GLiNER2

extractor = GLiNER2.from_pretrained("mlx_models/fastino_gliner2-base-v1")

result = extractor.extract_entities(
    "Apple CEO Tim Cook announced iPhone 15 in Cupertino.",
    ["company", "person", "product", "location"]
)
# {"company": ["Apple"], "person": ["Tim Cook"], "product": ["iPhone 15"], "location": ["Cupertino"]}
```

## Features

- **Entity Extraction** - Named entity recognition
- **Text Classification** - Single and multi-label
- **Structured JSON Extraction** - Parse structured data from text
- **Relation Extraction** - Extract relationships between entities
- **Multi-Task Schemas** - Combine all tasks in a single pass
- **Batch Processing** - Process multiple texts efficiently

No PyTorch or GPU required. Runs entirely on MLX.
