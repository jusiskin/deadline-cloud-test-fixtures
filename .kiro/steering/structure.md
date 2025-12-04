---
inclusion: always
---

# Project Structure

## Directory Layout

```
deadline-cloud-test-fixtures/
├── src/deadline_test_fixtures/     # Main package source
│   ├── __init__.py                 # Public API exports
│   ├── fixtures.py                 # Pytest fixtures
│   ├── models.py                   # Data models and dataclasses
│   ├── util.py                     # Utility functions
│   ├── job_attachment_manager.py   # Job attachments management
│   ├── pytest_hooks.py             # Pytest plugin hooks
│   ├── cloudformation/             # CloudFormation stack definitions
│   ├── containers/                 # Docker container implementations
│   └── deadline/                   # Deadline-specific modules
│       ├── client.py               # DeadlineClient wrapper
│       ├── resources.py            # Farm, Fleet, Queue resources
│       └── worker.py               # Worker implementations
├── test/                           # Test suite
│   ├── unit/                       # Unit tests
│   └── test_copyright_headers.py   # Copyright validation
├── pipeline/                       # CI/CD scripts
├── scripts/                        # Utility scripts
└── .semantic_release/              # Release automation templates
```

## Code Organization

### Public API (`__init__.py`)
All public classes and fixtures are explicitly exported in `__all__`. This is the contract for package consumers.

### Fixtures (`fixtures.py`)
- Session-scoped pytest fixtures for resource management
- Fixtures use environment variables for configuration
- Resources are created/destroyed using context managers for cleanup
- Fixture dependencies are explicit via pytest's dependency injection

### Models (`models.py`)
- Frozen dataclasses for immutability
- Type hints using `from __future__ import annotations`
- Platform-specific logic (Linux vs Windows) encapsulated in model methods

### Worker Implementations
- Abstract base: `DeadlineWorker`
- Concrete implementations: `EC2InstanceWorker`, `DockerContainerWorker`
- Platform-specific workers: `PosixInstanceBuildWorker`, `WindowsInstanceBuildWorker`

## Conventions

### Copyright Headers
Every `.py` and `.sh` file must include the copyright header:
```python
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
```

### Type Annotations
- Use `from __future__ import annotations` for forward references
- Type hints required on all public functions and methods
- Use `|` for union types (Python 3.10+ syntax)

### Dataclasses
- Prefer `@dataclass(frozen=True)` for immutability
- Use `InitVar` for parameters that don't become fields
- Use `__post_init__` for validation and computed fields

### Resource Management
- Use context managers (`contextmanager`, `ExitStack`) for cleanup
- Session-scoped fixtures for expensive resources
- Explicit cleanup in finally blocks or fixture teardown

### Environment Variables
- Document all environment variables in fixture docstrings
- Provide sensible defaults where possible
- Use `os.getenv()` with defaults, `os.environ[]` when required
