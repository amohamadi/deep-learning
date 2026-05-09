# Deep Learning From Scratch 

This repository is designed as a clean and simple learning environment for understanding core deep learning concepts from scratch.

## Project Structure

- `deeplearning/`
  - Core implementation modules for a simple neural network framework.
  - `layers/` contains layer implementations such as `Dense`, `Dropout`, and `ActivationLayer`.
- `examples/`
  - Standalone example scripts demonstrating usage and learning workflows.
  - Subfolders for specific topics:
    - `attention/`
    - `mnist/`
    - `nlp/`
    - `tokenization/`
    - `embedding/`

## Getting Started

Run examples from the project root using Python 3. For example:

```bash
python3 examples/xor.py
python3 examples/attention/attention_walkthrough.py
python3 examples/mnist/mnist_loader.py
python3 examples/nlp/base_code.py
python3 examples/tokenization/bpe.py
python3 examples/embedding/word2vec.py
```

## Package Usage

The core framework is exposed through the `deeplearning` package. Example imports:

```python
from deeplearning.network import Network
from deeplearning.layers import Dense, ActivationLayer
from deeplearning.activations import tanh, tanh_prime
from deeplearning.losses import mse, mse_prime
```

## Notes

- The repository is intentionally simple and educational.
- Use `python3` to run the example scripts.
- The `examples/` folder is the best starting point for exploring the project.

## Recommended Workflow

1. Study the core module implementations in `deeplearning/`.
2. Run example scripts in `examples/`.
3. Modify models or add new examples as you learn.
