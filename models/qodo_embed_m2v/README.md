---
base_model: Qodo/Qodo-Embed-1-1.5B
library_name: model2vec
license: other
license_name: qodoai-open-rail-m
license_link: LICENSE
model_name: Qodo-Embed-M-1-1.5B-M2V-Distilled
tags:
- sentence-transformers
- sentence-similarity
- feature-extraction
- transformers
- Qwen2
---

# Qodo-Embed-M-1-1.5B-M2V-Distilled

This project optimizes the Qodo-Embed-1-1.5B model using Model2Vec, reducing its size and dramatically improving inference speed while maintaining most of its performance capabilities.

## Overview

[Qodo-Embed-1-1.5B](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B) is a state-of-the-art code embedding model designed for retrieval tasks in the software development domain. While powerful, it can be resource-intensive for production use cases.

[Model2Vec](https://github.com/MinishLab/model2vec) is a technique to distill large sentence transformer models into small, fast static embedding models. This project applies Model2Vec to create an optimized version of Qodo-Embed-1-1.5B with the following benefits:

- **Smaller Size**: Reduces model size by a factor of 25x
- **Faster Inference**: Up to 112x faster inference
- **Low Resource Requirements**: Minimal memory footprint and dependencies
- **Maintains Performance**: Retains most of the original model's capabilities

## Model Information

- **Model Name**: Qodo-Embed-M-1-1.5B-M2V-Distilled
- **Original Model**: [Qodo-Embed-1-1.5B](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B)
- **Distillation Method**: [Model2Vec](https://github.com/MinishLab/model2vec)
- **Original Dimensions**: 1536
- **Distilled Dimensions**: 384
- **Explained Variance**: ~85%
- **Size Reduction**: 25.26x (from 5.9GB to 233MB)
- **Speed Improvement**: 112.14x faster

## Installation

First, ensure you have the required dependencies:

```bash
# Install the base package
uv add --group model2vec model2vec 'model2vec[distill]' sentence-transformers transformers

# Install additional dependencies for evaluation
uv add --group model2vec matplotlib psutil
```

## Usage

### Distillation

To create a distilled version of Qodo-Embed-1-1.5B:

```bash
python models/qodo_embed_m2v/distill.py --pca_dims 384
```

Options:
- `--model_name` - Source model name (default: "Qodo/Qodo-Embed-1-1.5B")
- `--output_dir` - Where to save the distilled model (default: "models/qodo_embed_m2v")
- `--pca_dims` - Dimensions for PCA reduction; smaller values create faster but less accurate models (default: 384)
- `--save_to_hub` - Push the model to HuggingFace Hub
- `--hub_model_id` - Model ID for HuggingFace Hub (required if saving to hub)
- `--skip_readme` - Skip generating README file (default: True)

### Evaluation

To evaluate the distilled model against the original:

```bash
python models/qodo_embed_m2v/evaluate.py
```

Options:
- `--original_model` - Original model name (default: "Qodo/Qodo-Embed-1-1.5B")
- `--distilled_model` - Path to the distilled model (default: "models/qodo_embed_m2v")
- `--output_dir` - Where to save evaluation results (default: "models/qodo_embed_m2v/evaluation")

## Example Code

```python
from model2vec import StaticModel
from sentence_transformers import SentenceTransformer
import time

# Sample code for embedding
code_samples = [
    "def process_data_stream(source_iterator):",
    "implement binary search tree",
    "how to handle memory efficient data streaming",
    """class LazyLoader:
        def __init__(self, source):
            self.generator = iter(source)
            self._cache = []"""
]

# Load original model
print("Loading original model...")
original_model = SentenceTransformer("Qodo/Qodo-Embed-1-1.5B")

# Load distilled model
print("Loading distilled model...")
distilled_model = StaticModel.from_pretrained("models/qodo_embed_m2v")

# Compare embedding speed
print("\nGenerating embeddings with original model...")
start = time.time()
original_embeddings = original_model.encode(code_samples)
original_time = time.time() - start
print(f"Original model took: {original_time:.4f} seconds")

print("\nGenerating embeddings with distilled model...")
start = time.time()
distilled_embeddings = distilled_model.encode(code_samples)
distilled_time = time.time() - start
print(f"Distilled model took: {distilled_time:.4f} seconds")
print(f"Speed improvement: {original_time/distilled_time:.2f}x faster")

print(f"\nOriginal embedding dimensions: {original_embeddings.shape}")
print(f"Distilled embedding dimensions: {distilled_embeddings.shape}")
```

## Results

The distilled model achieves:

- 25.26x reduction in model size (from 5.9GB to 233MB)
- 112.14x increase in inference speed
- 85.1% explained variance with PCA reduction to 384 dimensions

Detailed evaluation results, including similarity plots and performance metrics, are saved to the evaluation output directory.

## Project Structure

- `distill.py` - Script to create the distilled model
- `evaluate.py` - Script to compare performance with the original model
- `example.py` - Example usage of the distilled model
- `evaluation/` - Directory containing evaluation results and visualizations

## Acknowledgments

This project is built upon the following technologies:

- [Qodo-Embed-1-1.5B](https://huggingface.co/Qodo/Qodo-Embed-1-1.5B) - The original code embedding model developed by QodoAI
- [Model2Vec](https://github.com/MinishLab/model2vec) - The distillation technique used to optimize the model

## License

This model is licensed under the [QodoAI-Open-RAIL-M](https://www.qodo.ai/open-rail-m-license/) license, the same as the original Qodo-Embed-1-1.5B model. Any derivative model must include "Qodo" at the beginning of its name per the license requirements. 