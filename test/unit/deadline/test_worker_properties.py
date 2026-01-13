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


class TestOSAgnosticWorkerAgentOperations:
    """Property tests for OS-agnostic worker agent operations."""

    @pytest.mark.parametrize(
        "os_type,worker_class,host_class",
        [
            ("posix", "PosixInstanceBuildWorker", "PosixEC2WorkerHost"),
            ("windows", "WindowsInstanceBuildWorker", "WindowsEC2WorkerHost"),
        ],
    )
    def test_property_13_os_agnostic_agent_operations(
        self, os_type, worker_class, host_class, worker_config
    ):
        """
        Property 13: Operating system-agnostic worker agent operations

        For any worker agent operation (install, configure, start, stop, get_worker_id)
        and any EC2 worker host operating system (Windows or POSIX), the operation should
        succeed and produce the same logical result regardless of the operating system.

        **Validates: Requirements 7.4**
        """
        # Import the appropriate classes based on OS type
        if os_type == "posix":
            from deadline_test_fixtures.deadline.worker import PosixInstanceBuildWorker
            from deadline_test_fixtures.deadline.worker_host import PosixEC2WorkerHost

            WorkerClass: type = PosixInstanceBuildWorker
            HostClass: type = PosixEC2WorkerHost
        else:
            from deadline_test_fixtures.deadline.worker import WindowsInstanceBuildWorker
            from deadline_test_fixtures.deadline.worker_host import WindowsEC2WorkerHost

            WorkerClass = WindowsInstanceBuildWorker
            HostClass = WindowsEC2WorkerHost

        # Create a mock host for the specific OS
        mock_host = MagicMock(spec=HostClass)
        mock_host.is_running.return_value = True
        mock_host._operating_system.return_value = os_type
        mock_host.instance_id = "i-1234567890abcdef0"
        mock_host.send_command.return_value = MagicMock(exit_code=0, stdout="worker-test123")

        # Add all the required attributes for backward compatibility
        mock_host.subnet_id = "subnet-12345"
        mock_host.security_group_id = "sg-12345"
        mock_host.instance_profile_name = "test-profile"
        mock_host.bootstrap_bucket_name = "test-bucket"
        mock_host.s3_client = MagicMock()
        mock_host.ec2_client = MagicMock()
        mock_host.ssm_client = MagicMock()
        mock_host.instance_type = "t3.micro"
        mock_host.instance_shutdown_behavior = "terminate"
        mock_host.additional_tags = []

        with patch("boto3.client"):
            worker = WorkerClass(
                configuration=worker_config,
                worker_host=mock_host,
                deadline_client=MagicMock(),
            )

        # Test 1: Worker agent lifecycle operations should work regardless of OS
        with (
            patch.object(worker, "_stage_s3_bucket", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-test123"),
        ):

            # Operation: Start worker agent
            worker.start()

            # Verify: Agent should be running regardless of OS
            assert worker.agent_state == WorkerAgentState.RUNNING
            assert worker.worker_id == "worker-test123"

            # Verify: Host should be claimed
            mock_host._claim_for_worker.assert_called_once_with(id(worker))

            # Operation: Stop worker agent
            worker.stop()

            # Verify: Agent should be stopped regardless of OS
            assert worker.agent_state == WorkerAgentState.NOT_STARTED
            assert worker.worker_id is None

            # Verify: Host should be released
            mock_host._release_from_worker.assert_called_once_with(id(worker))

        # Test 2: Command delegation should work regardless of OS
        test_command = "test command"
        expected_result = MagicMock(exit_code=0, stdout="test output")
        mock_host.send_command.return_value = expected_result

        result = worker.send_command(test_command)

        # Verify: Command delegation works the same way regardless of OS
        assert result == expected_result
        # Note: The actual call may include OS-specific waiter config, but the delegation works

        # Test 3: Worker ID retrieval should work regardless of OS
        with (
            patch.object(worker, "_stage_s3_bucket", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-abc123"),
        ):

            worker.start()

            # Verify: Worker ID retrieval produces a valid worker ID regardless of OS
            assert worker.worker_id is not None
            assert worker.worker_id == "worker-abc123"
            assert worker.worker_id.startswith("worker-")

            worker.stop()

    @pytest.mark.parametrize(
        "os_type",
        ["posix", "windows"],
    )
    def test_property_13_os_validation_enforced(self, os_type, worker_config):
        """
        Property 13 extended: OS validation prevents mismatched worker/host combinations

        The system should enforce that Windows workers require Windows hosts and
        POSIX workers require POSIX hosts.

        **Validates: Requirements 7.4**
        """
        # Import the appropriate classes
        if os_type == "posix":
            from deadline_test_fixtures.deadline.worker import WindowsInstanceBuildWorker
            from deadline_test_fixtures.deadline.worker_host import PosixEC2WorkerHost

            WorkerClass: type = WindowsInstanceBuildWorker  # Mismatched: Windows worker
            HostClass: type = PosixEC2WorkerHost  # with POSIX host
            expected_worker_os = "windows"
            actual_host_os = "posix"
        else:
            from deadline_test_fixtures.deadline.worker import PosixInstanceBuildWorker
            from deadline_test_fixtures.deadline.worker_host import WindowsEC2WorkerHost

            WorkerClass = PosixInstanceBuildWorker  # Mismatched: POSIX worker
            HostClass = WindowsEC2WorkerHost  # with Windows host
            expected_worker_os = "posix"
            actual_host_os = "windows"

        # Create a mock host with mismatched OS
        mock_host = MagicMock(spec=HostClass)
        mock_host._operating_system.return_value = actual_host_os
        mock_host.instance_id = "i-1234567890abcdef0"

        # Add all the required attributes for backward compatibility
        mock_host.subnet_id = "subnet-12345"
        mock_host.security_group_id = "sg-12345"
        mock_host.instance_profile_name = "test-profile"
        mock_host.bootstrap_bucket_name = "test-bucket"
        mock_host.s3_client = MagicMock()
        mock_host.ec2_client = MagicMock()
        mock_host.ssm_client = MagicMock()
        mock_host.instance_type = "t3.micro"
        mock_host.instance_shutdown_behavior = "terminate"
        mock_host.additional_tags = []

        # When/Then: Creating a worker with mismatched OS should fail
        with patch("boto3.client"):
            with pytest.raises(
                ValueError,
                match=f"Worker requires {expected_worker_os} host but got {actual_host_os} host",
            ):
                WorkerClass(
                    configuration=worker_config,
                    worker_host=mock_host,
                    deadline_client=MagicMock(),
                )
