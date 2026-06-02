# Contributing to MSTS-AN

Thank you for your interest in contributing to MSTS-AN! This document provides guidelines and instructions for contributing to this project.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open an issue with the following information:
- Clear description of the bug
- Steps to reproduce
- Expected behavior
- Actual behavior
- Environment details (OS, Python version, GPU)
- Error messages or logs

### Suggesting Enhancements

Enhancement suggestions are welcome! Please:
- Use a clear and descriptive title
- Provide detailed description of the proposed feature
- Explain why this enhancement would be useful
- Include code examples if applicable

### Pull Requests

1. Fork the repository
2. Create a new branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and ensure code quality
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/MSTS-AN.git
cd MSTS-AN

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We follow PEP 8 style guidelines. Please ensure your code:
- Passes `flake8` linting
- Is formatted with `black`
- Has imports sorted with `isort`
- Includes type hints where appropriate
- Has comprehensive docstrings

### Formatting Commands

```bash
# Format code
black models/ data/ utils/ tests/
isort models/ data/ utils/ tests/

# Run linter
flake8 models/ data/ utils/ tests/

# Run type checker
mypy models/ data/ utils/
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=models --cov=data --cov=utils

# Run specific test
pytest tests/test_model.py::test_mstsan_forward
```

## Documentation

- Update README.md if adding new features
- Add docstrings to all public functions and classes
- Update CHANGELOG.md with your changes
- Add examples for new functionality

## Commit Messages

Use clear and meaningful commit messages:
- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and pull requests where appropriate

Example:
```
Add support for multi-GPU training

- Implement DataParallel wrapper
- Add distributed training script
- Update documentation with multi-GPU instructions

Fixes #123
```

## Code Review Process

1. Maintainers will review your PR within 5 business days
2. Address review comments and push updates
3. Once approved, a maintainer will merge your PR

## Questions?

Feel free to open an issue for:
- Questions about the codebase
- Clarification on contributing guidelines
- Discussion of potential features

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

Thank you for contributing to MSTS-AN!
