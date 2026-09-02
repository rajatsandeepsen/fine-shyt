# Fine-Shyt
uv template for QLoRA fine tuning models

## Requirements

- [Install `uv`](https://docs.astral.sh/uv/getting-started/installation/)
- A computer with CUDA-capable GPU

## Installation

```bash
uv sync
```

## Commands

### GPU check

```bash
uv run poe gpu
```

### Model config source

Modify [`src/fine_tune/config.py`](./src/fine_tune/config.py)

This is the main config source for fine-tuning defaults.

### Fine-tune model

```bash
uv run poe fine-tune
```

### Run inference script

```bash
uv run poe fine-infer
```

### Test template

```bash
uv run poe test-template
```

### Start dev inference server

```bash
uv run poe dev
```

### Start production inference server

```bash
uv run poe prod
```

### Start Jupyter Notebook Lab

```bash
uv run poe lab
```

## Dependencies

Core dependencies (from `pyproject.toml`):

- `datasets`
- `fastapi[standard]`
- `torch`
- `unsloth`

Dev dependencies:

- `ipykernel`
- `poethepoet`
- `ty`
