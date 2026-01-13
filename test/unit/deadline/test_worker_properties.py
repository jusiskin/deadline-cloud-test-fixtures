# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Property-based tests for worker host decoupling."""

import pytest
from unittest.mock import MagicMock, patch

from deadline_test_fixtures.deadline.worker import PosixInstanceBuildWorker
from deadline_test_fixtures.deadline.worker_host import (
    PosixEC2WorkerHost,
    WorkerAgentState,
)
from deadline_test_fixtures import (
    DeadlineWorkerConfiguration,
    Fleet,
    Farm,
    PipInstall,
    CodeArtifactRepositoryInfo,
)


@pytest.fixture
def mock_worker_host():
    """Create a mock worker host for testing."""
    host = MagicMock(spec=PosixEC2WorkerHost)
    host.is_running.return_value = True
    host._operating_system.return_value = "posix"
    host.instance_id = "i-1234567890abcdef0"
    host.send_command.return_value = MagicMock(exit_code=0, stdout="worker-test123")

    # Add all the required attributes for backward compatibility
    host.subnet_id = "subnet-12345"
    host.security_group_id = "sg-12345"
    host.instance_profile_name = "test-profile"
    host.bootstrap_bucket_name = "test-bucket"
    host.s3_client = MagicMock()
    host.ec2_client = MagicMock()
    host.ssm_client = MagicMock()
    host.instance_type = "t3.micro"
    host.instance_shutdown_behavior = "terminate"
    host.additional_tags = []

    return host


@pytest.fixture
def worker_config():
    """Create a test worker configuration."""
    return DeadlineWorkerConfiguration(
        farm_id="farm-123",
        fleet=Fleet(id="fleet_123", farm=Farm(id="farm-123")),
        region="us-west-2",
        job_user="test-user",
        job_user_group="test-group",
        allow_shutdown=False,
        worker_agent_install=PipInstall(
            requirement_specifiers=["deadline-cloud-worker-agent"],
            codeartifact=CodeArtifactRepositoryInfo(
                region="us-west-2",
                domain="test-domain",
                domain_owner="123456789123",
                repository="test-repository",
            ),
        ),
    )


@pytest.fixture
def worker(worker_config, mock_worker_host):
    """Create a test worker with mocked dependencies."""
    with patch("boto3.client"):
        worker = PosixInstanceBuildWorker(
            configuration=worker_config,
            worker_host=mock_worker_host,
            deadline_client=MagicMock(),
        )
        return worker


class TestWorkerAgentLifecycleProperties:
    """Property tests for worker agent lifecycle independence."""

    def test_property_6_worker_agent_stop_preserves_host(self, worker, mock_worker_host):
        """
        Property 6: Worker agent stop preserves host

        For any worker agent that is running, stopping the agent should leave the host
        in a running state and available for reuse by other workers.

        **Validates: Requirements 2.4**
        """
        # Given: A running worker host and agent
        mock_worker_host.is_running.return_value = True

        with (
            patch.object(worker, "_stage_s3_bucket", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-test123"),
        ):

            # Start the worker agent
            worker.start()
            assert worker.agent_state == WorkerAgentState.RUNNING
            assert worker.worker_id == "worker-test123"

            # When: Stop the worker agent
            worker.stop()

            # Then: Host should still be running and agent should be stopped
            assert worker.agent_state == WorkerAgentState.NOT_STARTED
            assert worker.worker_id is None
            # Host should still be running (not stopped)
            mock_worker_host.is_running.assert_called()
            # Host should be released from this worker
            mock_worker_host._release_from_worker.assert_called_with(id(worker))

    def test_property_7_method_delegation_correctness(self, worker, mock_worker_host):
        """
        Property 7: Method delegation correctness

        For any command sent through the worker, it should be correctly delegated
        to the worker host without modification.

        **Validates: Requirements 4.4, 4.2**
        """
        # Given: A command to send
        test_command = "echo 'test command'"
        expected_result = MagicMock(exit_code=0, stdout="test output")
        mock_worker_host.send_command.return_value = expected_result

        # When: Send command through worker
        result = worker.send_command(test_command)

        # Then: Command should be delegated to host (with default waiter config for POSIX)
        mock_worker_host.send_command.assert_called_once_with(
            test_command, {"Delay": 5, "MaxAttempts": 30}
        )
        assert result == expected_result

    def test_property_7_command_delegation_with_various_commands(self, worker, mock_worker_host):
        """
        Property 7 extended: Method delegation works for various command strings

        For any valid command string, delegation to worker host should work correctly.
        """
        # Test with various command strings
        test_commands = [
            "echo hello",
            "ls -la",
            "cat /etc/passwd",
            "ps aux | grep python",
            "find /tmp -name '*.log'",
        ]

        for command in test_commands:
            # Given: Various command strings
            expected_result = MagicMock(exit_code=0, stdout="output")
            mock_worker_host.send_command.return_value = expected_result

            # When: Send command through worker
            result = worker.send_command(command)

            # Then: Command should be delegated exactly (with default waiter config for POSIX)
            mock_worker_host.send_command.assert_called_with(
                command, {"Delay": 5, "MaxAttempts": 30}
            )
            assert result == expected_result

            # Reset for next iteration
            mock_worker_host.send_command.reset_mock()

    def test_host_ownership_prevents_concurrent_agents(self, worker_config, mock_worker_host):
        """
        Test that host ownership prevents multiple workers from running agents simultaneously.

        **Validates: Requirements 2.4**
        """
        with patch("boto3.client"):
            # Create two workers sharing the same host
            worker1 = PosixInstanceBuildWorker(
                configuration=worker_config,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )
            worker2 = PosixInstanceBuildWorker(
                configuration=worker_config,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

        # Mock the host claiming behavior
        active_worker_id = None

        def mock_claim_for_worker(worker_id):
            nonlocal active_worker_id
            if active_worker_id is not None and active_worker_id != worker_id:
                raise RuntimeError(
                    f"Another worker (id={active_worker_id}) already has an agent running"
                )
            active_worker_id = worker_id

        def mock_release_from_worker(worker_id):
            nonlocal active_worker_id
            if active_worker_id == worker_id:
                active_worker_id = None

        mock_worker_host._claim_for_worker.side_effect = mock_claim_for_worker
        mock_worker_host._release_from_worker.side_effect = mock_release_from_worker

        with (
            patch.object(worker1, "_stage_s3_bucket", return_value=None),
            patch.object(worker1, "get_worker_id", return_value="worker-test123"),
            patch.object(worker2, "_stage_s3_bucket", return_value=None),
            patch.object(worker2, "get_worker_id", return_value="worker-test456"),
        ):

            # Start first worker - should succeed
            worker1.start()
            assert worker1.agent_state == WorkerAgentState.RUNNING

            # Try to start second worker - should fail
            with pytest.raises(RuntimeError, match="Another worker.*already has an agent running"):
                worker2.start()

            # Stop first worker
            worker1.stop()
            assert worker1.agent_state == WorkerAgentState.NOT_STARTED

            # Now second worker should be able to start
            worker2.start()
            assert worker2.agent_state == WorkerAgentState.RUNNING

    def test_worker_requires_running_host(self, worker, mock_worker_host):
        """
        Test that worker agent cannot start if host is not running.

        **Validates: Requirements 2.4**
        """
        # Given: A non-running host
        mock_worker_host.is_running.return_value = False

        # When/Then: Starting worker should fail
        with pytest.raises(
            RuntimeError, match="Cannot start worker agent: worker host is not running"
        ):
            worker.start()

    def test_worker_cannot_start_twice(self, worker, mock_worker_host):
        """
        Test that worker agent cannot be started twice without stopping first.

        **Validates: Requirements 2.4**
        """
        with (
            patch.object(worker, "_stage_s3_bucket", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-test123"),
        ):

            # Start worker once
            worker.start()
            assert worker.agent_state == WorkerAgentState.RUNNING

            # Try to start again - should fail
            with pytest.raises(
                RuntimeError,
                match="Cannot start worker agent: this worker already has an agent running",
            ):
                worker.start()
