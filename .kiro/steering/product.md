---
inclusion: always
---

# Product Overview

AWS Deadline Cloud Test Fixtures is both a pytest plugin and a Python library that provides reusable test fixtures and utilities for testing AWS Deadline Cloud Python packages.

## Purpose

- Provides pytest fixtures for integration and end-to-end testing of Deadline Cloud components
- Acts as a Python library for downstream projects to build custom test fixtures
- Manages test infrastructure including EC2 workers, Docker containers, and AWS resources
- Handles deployment and cleanup of test resources (Farms, Fleets, Queues)
- Supports both Amazon Linux (AL2023) and Windows (WIN2022) worker environments
- Enables Deadline Cloud job submission and monitoring including log checking

## Key Components

- **DeadlineWorker**: Abstract interface for worker implementations (EC2, Docker)
- **DeadlineClient**: Wrapper for boto3 Deadline API client
- **Resource Management**: Farm, Fleet, Queue, and Job Attachment resources
- **Bootstrap Resources**: IAM roles, S3 buckets, and CloudFormation stacks for test infrastructure
- **Job Management**: Classes for submitting jobs and monitoring their execution (Job, Session, Step, Task)
- **Log Monitoring**: CloudWatch log event handling and session log retrieval

## Usage Patterns

### As a pytest plugin
Tests import fixtures from this package to get pre-configured Deadline resources and workers. The package handles the complexity of resource provisioning, configuration, and cleanup.

### As a Python library
Downstream projects can import classes directly to:
- Build custom pytest fixtures tailored to their testing needs
- Submit Deadline Cloud jobs programmatically
- Monitor job execution and task status
- Retrieve and validate CloudWatch logs and session logs
