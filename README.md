# MLX LRASPP Mobilenet V3 Large

Conversion of the LRASPP Mobilenet V3 Large PyTorch segmentation model to MLX.

## Installation

```shell
uv sync
```

## Conversion

This script convert the PyTorch model weights to MLX safetensors weights compatible with the MLX model architecture defined in the `modeling.py` file.

```shell
uv run convert.py
```

## Validation

This script checks that the MLX model outputs a value very close to the PyTorch model (random input) and benchmarks the inference speed.

```shell
uv run validation.py
```
