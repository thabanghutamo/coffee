# Contributing to Vocal MIDI Generator

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Ways to Contribute

- 🐛 **Bug Reports**: Found a bug? Open an issue!
- ✨ **Feature Requests**: Have an idea? We'd love to hear it!
- 📝 **Documentation**: Improve our docs
- 🎨 **UI/UX**: Design improvements
- 🧪 **Testing**: Write tests, test features
- 🎵 **Presets**: Share genre templates or MIDI patterns
- 🤖 **ML Models**: Improve model architectures or training

## Getting Started

1. **Fork the repository**
2. **Clone your fork**:
   ```bash
   git clone https://github.com/YOUR_USERNAME/vocal-midi-generator.git
   cd vocal-midi-generator
   ```
3. **Run setup**:
   ```bash
   ./setup.sh
   ```
4. **Create a branch**:
   ```bash
   git checkout -b feature/my-awesome-feature
   ```

## Development Workflow

### For C++ Plugin Code

1. Make changes in `src/`
2. Build and test:
   ```bash
   cd build
   cmake --build . -j8
   ./VocalMIDI_artefacts/Release/Standalone/VocalMIDI
   ```
3. Run in a DAW to test integration

### For ML Models

1. Make changes in `ml_training/models/`
2. Test models:
   ```bash
   cd ml_training
   source venv/bin/activate
   python -m pytest tests/  # if tests exist
   python -c "from models import YourModel; YourModel()"
   ```
3. Export to ONNX and test in plugin

### For Documentation

1. Edit markdown files in `docs/` or root
2. Preview locally
3. Check for broken links

## Code Style

### C++ Code
- Follow JUCE coding conventions
- Use `camelCase` for methods
- Use `PascalCase` for classes
- Use descriptive variable names
- Add comments for complex logic

Example:
```cpp
class MyAwesomeClass
{
public:
    void doSomethingCool(int parameter)
    {
        // Clear explanation of what this does
        myMemberVariable = parameter * 2;
    }
    
private:
    int myMemberVariable;
};
```

### Python Code
- Follow PEP 8
- Use `snake_case` for functions and variables
- Use `PascalCase` for classes
- Add docstrings

Example:
```python
class MyModel(nn.Module):
    """
    Brief description of the model.
    
    Args:
        input_dim: Size of input
        hidden_dim: Size of hidden layer
    """
    
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
```

## Commit Messages

Use clear, descriptive commit messages:

```
Add feature: Real-time beat synchronization

- Implement beat detection algorithm
- Add sync to DAW transport
- Update UI with beat indicators
```

Format: `<type>: <description>`

Types:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, no code change
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

## Pull Request Process

1. **Update documentation** if needed
2. **Add tests** for new features
3. **Ensure builds pass** on all platforms
4. **Update CHANGELOG.md** (if exists)
5. **Request review** from maintainers

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement

## Testing
How was this tested?

## Screenshots (if applicable)
Visual changes

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Tests added/updated
```

## Testing Guidelines

### Manual Testing
- Test in multiple DAWs (Ableton, Logic, FL Studio)
- Test on different platforms (Win/Mac/Linux)
- Test with different audio interfaces
- Test edge cases (extreme BPM, unusual input)

### Automated Testing
- Write unit tests for algorithms
- Test model inference
- Benchmark latency

## Building for Different Platforms

### Linux
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

### macOS
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

### Windows
```bash
cmake -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
```

## Issue Guidelines

When opening an issue:

1. **Search first**: Check if already reported
2. **Use template**: Fill out issue template
3. **Be specific**: Include steps to reproduce
4. **Include details**:
   - OS and version
   - DAW and version
   - Plugin version
   - Console logs
   - Screenshots/videos if relevant

### Bug Report Template

```markdown
**Describe the bug**
Clear description

**To Reproduce**
1. Load plugin in [DAW]
2. Set BPM to [value]
3. Sing/hum
4. See error

**Expected behavior**
What should happen

**Screenshots**
If applicable

**Environment:**
- OS: [e.g., macOS 13.0]
- DAW: [e.g., Ableton Live 11]
- Plugin Version: [e.g., 1.0.0]

**Additional context**
Any other info
```

## ML Model Contributions

If contributing ML improvements:

1. **Document architecture changes** in `docs/ML_ARCHITECTURE.md`
2. **Provide training scripts** or update existing ones
3. **Share performance metrics**:
   - Accuracy
   - Latency
   - Memory usage
4. **Include dataset requirements**
5. **Export to ONNX** for C++ integration

## Documentation Contributions

- Fix typos and unclear explanations
- Add examples and tutorials
- Improve API documentation
- Create video tutorials
- Translate to other languages

## Community

- Be respectful and welcoming
- Help others in discussions
- Share your creations made with the plugin
- Provide constructive feedback

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Open a Discussion on GitHub or reach out to maintainers.

---

**Thank you for contributing! 🎉**
