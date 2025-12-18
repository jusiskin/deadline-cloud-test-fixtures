# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
from __future__ import annotations

import abc
from enum import Enum
from typing import Optional

from ..deadline.worker import CommandResult


class WorkerHostState(Enum):
    """Worker host lifecycle states."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    STOPPED = "stopped"


class WorkerAgentState(Enum):
    """Worker agent lifecycle states."""

    NOT_STARTED = "not_started"
    RUNNING = "running"
    STOPPED = "stopped"


class WorkerHost(abc.ABC):
    """Abstract base class for worker host management."""

    def __init__(self):
        self._state = WorkerHostState.NOT_STARTED
        self._active_worker_id: Optional[int] = None  # Track which worker is using this host

    @property
    def state(self) -> WorkerHostState:
        """Get the current state of the worker host."""
        return self._state

    @property
    def has_active_worker(self) -> bool:
        """Check if a worker agent is currently active on this host."""
        return self._active_worker_id is not None

    @abc.abstractmethod
    def _operating_system(self) -> str:
        """Return the operating system identifier (e.g., 'windows', 'posix')."""
        pass

    def start(self) -> None:
        """Start the worker host."""
        if self._state == WorkerHostState.RUNNING:
            raise RuntimeError("Worker host is already running")
        if self._state == WorkerHostState.STOPPED:
            raise RuntimeError("Cannot restart a stopped worker host")

        self._do_start()
        self._state = WorkerHostState.RUNNING

    @abc.abstractmethod
    def _do_start(self) -> None:
        """Implementation-specific start logic."""
        pass

    def stop(self) -> None:
        """Stop the worker host and clean up resources.

        This method can be called from any state except STOPPED:
        - From RUNNING: Stops the running host and cleans up resources
        - From NOT_STARTED: Cleans up any partial resources from a failed start attempt
        """
        if self._state == WorkerHostState.STOPPED:
            raise RuntimeError("Worker host is already stopped")

        # Always call _do_stop() to clean up resources, even if start() failed partway
        self._do_stop()
        self._state = WorkerHostState.STOPPED

    @abc.abstractmethod
    def _do_stop(self) -> None:
        """Implementation-specific stop logic."""
        pass

    @abc.abstractmethod
    def send_command(self, command: str) -> CommandResult:
        """Send a command to the worker host and return the result."""
        pass

    def is_running(self) -> bool:
        """Check if the worker host is currently running."""
        return self._state == WorkerHostState.RUNNING

    def _claim_for_worker(self, worker_id: int) -> None:
        """
        Claim this host for a worker agent.

        Raises:
            RuntimeError: If another worker already has an agent on this host
        """
        if self._active_worker_id is not None and self._active_worker_id != worker_id:
            raise RuntimeError(
                f"Cannot start worker agent: another worker (id={self._active_worker_id}) "
                f"already has an agent running on this host. "
                f"Call stop() on that worker first."
            )
        self._active_worker_id = worker_id

    def _release_from_worker(self, worker_id: int) -> None:
        """Release this host from a worker agent."""
        if self._active_worker_id == worker_id:
            self._active_worker_id = None
