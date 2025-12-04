---
inclusion: always
---

# Technology Stack

## Build System

- **Build Tool**: Hatch (Python project manager)
- **Version Control**: hatch-vcs for version management from git tags
- **Python Versions**: 3.9, 3.10, 3.11, 3.12

## Core Dependencies

- **boto3** (>=1.34.75, <2.0): AWS SDK for Python
- **botocore** (>=1.34.75): Core functionality for boto3
- **pytest**: Testing framework (this package is a pytest plugin)

## Development Tools

- **mypy**: Static type checking
- **ruff**: Fast Python linter and formatter
- **black**: Code formatter (line-length: 100)
- **pytest-cov**: Code coverage reporting
- **pytest-xdist**: Parallel test execution

## Common Commands

The CI/CD runs linting, build, and unit tests which all need to pass before a pull request can be merged. These should all be passing before creating a commit.
Formatting should be run before linting to fix style issues before they fail during lint. Building should be ran to create the _version.py file. Otherwise
an import error may happen when running tests. This is only needed if src/deadline_test_fixtures/_version.py is not present.

### Code Quality
```bash
# Run formatting
hatch run fmt

# Run linting
hatch run lint
```

### Build
```bash
hatch build
```

### Testing
```bash
# Run tests
hatch run test

# Run tests for all Python versions
hatch run all:test
```

## Configuration Notes

- Line length: 100 characters (enforced by ruff and black)
- Type checking: enabled with `check_untyped_defs = true`
- Test coverage: Reports generated in `build/coverage/`
- Parallel testing: Uses `--numprocesses=auto` by default
