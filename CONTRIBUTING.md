# Contributing to Swahili-100

Thank you for your interest in contributing to Swahili-100.

## How to Contribute

### 1. Find an issue or create one

Check the [issues page](https://github.com/marcoharuni/swahili-100/issues) for open tasks.

Priority areas:
- **Data sourcing** — Finding and cleaning more Swahili text
- **Tokenizer optimization** — Improving compression ratio
- **Model implementation** — Implementing the from-scratch components
- **Evaluation** — Adding benchmarks and baselines
- **Documentation** — Improving guides and translating to Swahili

### 2. Fork and branch

```bash
git fork https://github.com/marcoharuni/swahili-100.git
git checkout -b feature/your-feature-name
```

### 3. Make your changes

- Follow the existing code style
- Add docstrings to all public functions
- Write tests for new functionality
- Update documentation if needed

### 4. Test

```bash
python -m pytest tests/ -v
```

### 5. Submit a pull request

- Describe what you changed and why
- Reference any related issues
- Include test results

## Code Style

- Python 3.10+ type hints
- Docstrings for all public functions and classes
- No external ML library imports for core components (the point is from-scratch)
- PyTorch is used only for tensor operations and autograd

## Questions?

Open an issue or reach out on X.
