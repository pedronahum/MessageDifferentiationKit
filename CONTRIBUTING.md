# Contributing to MessageDifferentiationKit

Thank you for your interest in contributing to MessageDifferentiationKit! This document provides guidelines and instructions for contributing to the project.

## Code of Conduct

By participating in this project, you agree to maintain a respectful and inclusive environment for all contributors.

## Getting Started

### Prerequisites

- Swift 5.9 or later
- MPI 5.0 compliant implementation (OpenMPI 5.0+ or MPICH 4.2+)
- Familiarity with MPI concepts
- Understanding of automatic differentiation (helpful but not required)

### Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/yourusername/MessageDifferentiationKit.git
cd MessageDifferentiationKit
```

2. Install MPI:
```bash
# macOS
brew install open-mpi

# Linux
sudo apt-get install libopenmpi-dev
```

3. Build the project:
```bash
swift build
```

4. Run tests:
```bash
swift test
```

## Development Workflow

### Branching Strategy

- `main`: Stable releases
- `develop`: Active development
- `feature/*`: New features
- `bugfix/*`: Bug fixes
- `docs/*`: Documentation updates

### Making Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the coding guidelines below

3. Write or update tests for your changes

4. Ensure all tests pass:
```bash
swift test
```

5. Commit your changes with a descriptive message:
```bash
git commit -m "Add feature: description of your changes"
```

6. Push to your fork and create a Pull Request

## Coding Guidelines

### Swift Style

- Follow [Swift API Design Guidelines](https://swift.org/documentation/api-design-guidelines/)
- Use descriptive variable and function names
- Add documentation comments for public APIs
- Keep functions focused and concise

### MPI Bindings

- Always check error codes with `checkMPIError()`
- Use type-safe wrappers over raw C calls
- Provide Swift-idiomatic APIs
- Document MPI 5.0 specific features clearly

### Automatic Differentiation

- Mark functions with `@differentiable` where appropriate
- Ensure gradient correctness with tests
- Document any limitations of AD support

### Example

```swift
/// Performs an MPI broadcast operation
/// - Parameters:
///   - value: The value to broadcast
///   - root: Root process rank
///   - comm: MPI communicator
/// - Returns: The broadcasted value
/// - Throws: MPIError if the operation fails
@differentiable
public func broadcast<T: MPIDataRepresentable>(
    _ value: T,
    root: Int32,
    comm: MPI5.Communicator
) throws -> T {
    // Implementation
}
```

## Testing

### Unit Tests

- Write unit tests for all new functionality
- Place tests in the appropriate test target:
  - `MPISwiftTests/`: Low-level MPI binding tests
  - `MessageDifferentiationKitTests/`: High-level API tests
  - `IntegrationTests/`: End-to-end tests requiring MPI runtime

### Running Tests

```bash
# All tests
swift test

# Specific test
swift test --filter testMPIDataTypes

# With parallel execution (for MPI tests)
mpirun -np 4 swift test
```

### Test Coverage

We aim for 80%+ code coverage. Check coverage with:
```bash
swift test --enable-code-coverage
```

## Documentation

### API Documentation

- Use Swift DocC format for documentation comments
- Include:
  - Summary description
  - Parameters
  - Return values
  - Thrown errors
  - Example usage when helpful

### User Guides

- Add tutorials to `Documentation/` directory
- Include code examples
- Explain MPI 5.0 concepts when relevant

### Generating Documentation

```bash
swift package generate-documentation --target MessageDifferentiationKit
```

## Project Structure

When adding new files, follow the existing structure:

```
Sources/
├── CMPIBindings/         # C module for MPI headers
├── MPISwift/             # Low-level Swift MPI bindings
└── MessageDifferentiationKit/  # High-level differentiable API

Tests/
├── MPISwiftTests/
├── MessageDifferentiationKitTests/
└── IntegrationTests/
```

## Pull Request Process

1. Update the README.md if adding new features
2. Update documentation and examples
3. Ensure all tests pass
4. Update the CHANGELOG.md (if applicable)
5. Request review from maintainers

### PR Guidelines

- Keep PRs focused on a single feature/fix
- Write clear PR descriptions
- Link related issues
- Respond to review feedback promptly

## Reporting Issues

### Bug Reports

Include:
- Swift version
- MPI implementation and version
- Operating system
- Steps to reproduce
- Expected vs actual behavior
- Relevant code snippets

### Feature Requests

Include:
- Use case description
- Proposed API (if applicable)
- Benefits to the project
- Implementation ideas (optional)

## MPI 5.0 Compliance

When implementing MPI features:

1. Check MPI 5.0 standard specification
2. Use feature detection for optional capabilities
3. Provide fallbacks when possible
4. Document version requirements clearly

## Performance Considerations

- Profile performance-critical code
- Add benchmarks to `Benchmarks/` directory
- Document performance characteristics
- Compare with existing implementations (e.g., MeDiPack)

## Getting Help

- Open an issue for questions
- Check existing documentation
- Review the MPI 5.0 specification
- Consult Swift differentiable programming docs

## Recognition

Contributors will be acknowledged in:
- CONTRIBUTORS.md file
- Release notes
- Project documentation

Thank you for contributing to MessageDifferentiationKit!
