# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""Property-based tests for worker host decoupling."""

import pytest
from typing import Any
from unittest.mock import MagicMock, patch

from deadline_test_fixtures.deadline.worker import PosixInstanceBuildWorker, WorkerAgentError
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
            patch.object(worker, "_transfer_files", return_value=None),
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
            patch.object(worker1, "_transfer_files", return_value=None),
            patch.object(worker1, "get_worker_id", return_value="worker-test123"),
            patch.object(worker2, "_transfer_files", return_value=None),
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
            WorkerAgentError, match="Cannot start worker agent: worker host is not running"
        ):
            worker.start()

    def test_worker_cannot_start_twice(self, worker, mock_worker_host):
        """
        Test that worker agent cannot be started twice without stopping first.

        **Validates: Requirements 2.4**
        """
        with (
            patch.object(worker, "_transfer_files", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-test123"),
        ):

            # Start worker once
            worker.start()
            assert worker.agent_state == WorkerAgentState.RUNNING

            # Try to start again - should fail
            with pytest.raises(
                WorkerAgentError,
                match="Cannot start worker agent: this worker already has an agent running",
            ):
                worker.start()


class TestSequentialWorkerAgentConfiguration:
    """Property tests for sequential worker agent configuration."""

    def test_property_3_sequential_worker_agent_configuration(
        self, worker_config, mock_worker_host
    ):
        """
        Property 3: Sequential worker agent configuration

        For any worker host and any two different worker agent configurations,
        applying configuration A, stopping the agent, then applying configuration B
        should result in a working worker agent with configuration B's settings
        and no remnants of configuration A.

        **Validates: Requirements 1.2, 1.4, 2.5, 5.4**
        """
        # Create two different configurations
        config_a = worker_config
        config_b = DeadlineWorkerConfiguration(
            farm_id="farm-456",  # Different farm
            fleet=Fleet(id="fleet_456", farm=Farm(id="farm-456")),  # Different fleet
            region="us-east-1",  # Different region
            job_user="different-user",  # Different user
            job_user_group="different-group",  # Different group
            allow_shutdown=True,  # Different setting
            worker_agent_install=PipInstall(
                requirement_specifiers=["deadline-cloud-worker-agent==2.0.0"],  # Different version
                codeartifact=CodeArtifactRepositoryInfo(
                    region="us-east-1",
                    domain="different-domain",
                    domain_owner="987654321987",
                    repository="different-repository",
                ),
            ),
        )

        with patch("boto3.client"):
            # Create first worker with config A
            worker_a = PosixInstanceBuildWorker(
                configuration=config_a,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

            # Create second worker with config B (same host)
            worker_b = PosixInstanceBuildWorker(
                configuration=config_b,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

        with (
            patch.object(worker_a, "_transfer_files", return_value=None),
            patch.object(worker_a, "get_worker_id", return_value="worker-aaa123"),
            patch.object(worker_b, "_transfer_files", return_value=None),
            patch.object(worker_b, "get_worker_id", return_value="worker-bbb456"),
        ):

            # Apply configuration A
            worker_a.start()
            assert worker_a.agent_state == WorkerAgentState.RUNNING
            assert worker_a.worker_id == "worker-aaa123"
            assert worker_a.configuration.farm_id == "farm-123"
            assert worker_a.configuration.fleet.id == "fleet_123"

            # Stop agent with configuration A
            worker_a.stop()
            assert worker_a.agent_state == WorkerAgentState.NOT_STARTED
            assert worker_a.worker_id is None

            # Verify host is released and available for reuse
            mock_worker_host._release_from_worker.assert_called_with(id(worker_a))

            # Apply configuration B to the same host
            worker_b.start()
            assert worker_b.agent_state == WorkerAgentState.RUNNING
            assert worker_b.worker_id == "worker-bbb456"
            assert worker_b.configuration.farm_id == "farm-456"
            assert worker_b.configuration.fleet.id == "fleet_456"

            # Verify configuration B is completely independent from A
            assert worker_b.worker_id != worker_a.worker_id
            assert worker_b.configuration.farm_id != worker_a.configuration.farm_id
            assert worker_b.configuration.fleet.id != worker_a.configuration.fleet.id
            assert worker_b.configuration.region != worker_a.configuration.region

            # Clean up
            worker_b.stop()

    @pytest.mark.parametrize(
        "config_changes",
        [
            # Different farm and fleet
            {"farm_id": "farm-999", "fleet_id": "fleet_999"},
            # Different region
            {"region": "eu-west-1"},
            # Different user settings
            {"job_user": "custom-user", "job_user_group": "custom-group"},
            # Different shutdown setting
            {"allow_shutdown": True},
        ],
    )
    def test_property_3_various_configuration_changes(
        self, worker_config, mock_worker_host, config_changes
    ):
        """
        Property 3 extended: Sequential configuration with various changes

        Test that sequential configuration works with different types of configuration changes.

        **Validates: Requirements 1.2, 1.4, 2.5, 5.4**
        """
        # Create modified configuration
        config_b_params = {
            "farm_id": worker_config.farm_id,
            "fleet": worker_config.fleet,
            "region": worker_config.region,
            "job_user": worker_config.job_user,
            "job_user_group": worker_config.job_user_group,
            "allow_shutdown": worker_config.allow_shutdown,
            "worker_agent_install": worker_config.worker_agent_install,
        }

        # Apply the specific changes
        if "farm_id" in config_changes:
            config_b_params["farm_id"] = config_changes["farm_id"]
            config_b_params["fleet"] = Fleet(
                id=config_changes.get("fleet_id", "fleet_999"),
                farm=Farm(id=config_changes["farm_id"]),
            )
        if "region" in config_changes:
            config_b_params["region"] = config_changes["region"]
        if "job_user" in config_changes:
            config_b_params["job_user"] = config_changes["job_user"]
        if "job_user_group" in config_changes:
            config_b_params["job_user_group"] = config_changes["job_user_group"]
        if "allow_shutdown" in config_changes:
            config_b_params["allow_shutdown"] = config_changes["allow_shutdown"]

        config_b = DeadlineWorkerConfiguration(**config_b_params)

        with patch("boto3.client"):
            worker_a = PosixInstanceBuildWorker(
                configuration=worker_config,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )
            worker_b = PosixInstanceBuildWorker(
                configuration=config_b,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

        with (
            patch.object(worker_a, "_transfer_files", return_value=None),
            patch.object(worker_a, "get_worker_id", return_value="worker-aaa123"),
            patch.object(worker_b, "_transfer_files", return_value=None),
            patch.object(worker_b, "get_worker_id", return_value="worker-bbb456"),
        ):

            # Apply first configuration
            worker_a.start()
            assert worker_a.agent_state == WorkerAgentState.RUNNING

            # Stop and apply second configuration
            worker_a.stop()
            worker_b.start()
            assert worker_b.agent_state == WorkerAgentState.RUNNING

            # Verify the changed configuration is applied
            for key, value in config_changes.items():
                if key == "fleet_id":
                    assert worker_b.configuration.fleet.id == value
                elif key == "farm_id":
                    assert worker_b.configuration.farm_id == value
                elif key in ["region", "job_user", "job_user_group", "allow_shutdown"]:
                    assert getattr(worker_b.configuration, key) == value

            worker_b.stop()


class TestWorkerAgentConfigurationCorrectness:
    """Property tests for worker agent configuration correctness."""

    def test_property_4_worker_agent_configuration_correctness(
        self, worker_config, mock_worker_host
    ):
        """
        Property 4: Worker agent configuration correctness

        For any valid DeadlineWorkerConfiguration and any worker host, applying the
        configuration should result in a worker agent with settings that match the
        configuration (farm ID, fleet ID, region, user settings, etc.).

        **Validates: Requirements 1.3, 4.3, 5.3**
        """
        with patch("boto3.client"):
            worker = PosixInstanceBuildWorker(
                configuration=worker_config,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

        with (
            patch.object(worker, "_transfer_files", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-test123"),
        ):

            # Apply the configuration
            worker.start()

            # Verify: Worker agent should have the correct configuration
            assert worker.agent_state == WorkerAgentState.RUNNING
            assert worker.worker_id == "worker-test123"

            # Verify configuration matches
            assert worker.configuration.farm_id == "farm-123"
            assert worker.configuration.fleet.id == "fleet_123"
            assert worker.configuration.region == "us-west-2"
            assert worker.configuration.job_user == "test-user"
            assert worker.configuration.job_user_group == "test-group"
            assert worker.configuration.allow_shutdown is False

            # Verify the configure command was called with the correct configuration
            # The configure_worker_command method should be called during _configure_agent
            configure_cmd = worker.configure_worker_command(config=worker_config)

            # Verify the command contains the expected configuration values
            assert "farm-123" in configure_cmd
            assert "fleet_123" in configure_cmd
            assert "us-west-2" in configure_cmd
            # Note: agent_user (deadline-worker) is used for --user, not job_user
            assert "deadline-worker" in configure_cmd
            assert "test-group" in configure_cmd

            worker.stop()

    @pytest.mark.parametrize(
        "farm_id,fleet_id,region,job_user,allow_shutdown",
        [
            # Standard configuration
            ("farm-001", "fleet-001", "us-west-2", "user1", True),
            # Different region
            ("farm-002", "fleet-002", "eu-west-1", "user2", False),
            # Different user
            ("farm-003", "fleet-003", "us-east-1", "custom-user", True),
            # Edge case: Long IDs
            ("farm-" + "a" * 50, "fleet-" + "b" * 50, "ap-southeast-1", "user3", False),
            # Edge case: Special characters in user
            ("farm-004", "fleet-004", "us-west-1", "job-user-123", True),
        ],
    )
    def test_property_4_various_configurations(
        self,
        farm_id,
        fleet_id,
        region,
        job_user,
        allow_shutdown,
        mock_worker_host,
    ):
        """
        Property 4 extended: Configuration correctness with various valid configurations

        Test that configuration correctness holds for various valid configuration values.

        **Validates: Requirements 1.3, 4.3, 5.3**
        """
        # Create configuration with specific values
        config = DeadlineWorkerConfiguration(
            farm_id=farm_id,
            fleet=Fleet(id=fleet_id, farm=Farm(id=farm_id)),
            region=region,
            job_user=job_user,
            job_user_group="test-group",
            allow_shutdown=allow_shutdown,
            worker_agent_install=PipInstall(
                requirement_specifiers=["deadline-cloud-worker-agent"],
                codeartifact=CodeArtifactRepositoryInfo(
                    region=region,
                    domain="test-domain",
                    domain_owner="123456789123",
                    repository="test-repository",
                ),
            ),
        )

        with patch("boto3.client"):
            worker = PosixInstanceBuildWorker(
                configuration=config,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

        with (
            patch.object(worker, "_transfer_files", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-test123"),
        ):

            # Apply the configuration
            worker.start()

            # Verify: Configuration values match exactly
            assert worker.configuration.farm_id == farm_id
            assert worker.configuration.fleet.id == fleet_id
            assert worker.configuration.region == region
            assert worker.configuration.job_user == job_user
            assert worker.configuration.allow_shutdown == allow_shutdown

            # Verify the configure command contains the expected values
            configure_cmd = worker.configure_worker_command(config=config)
            assert farm_id in configure_cmd
            assert fleet_id in configure_cmd
            assert region in configure_cmd
            # Note: agent_user (deadline-worker) is used for --user, not job_user
            # job_user is used for job execution, not agent configuration
            assert "deadline-worker" in configure_cmd

            worker.stop()

    def test_property_4_configuration_with_file_mappings(self, worker_config, mock_worker_host):
        """
        Property 4 extended: Configuration correctness with file mappings

        Test that file mappings are correctly handled in the configuration.

        **Validates: Requirements 1.3, 4.3, 5.3**
        """
        # Create configuration with file mappings
        config_with_files = DeadlineWorkerConfiguration(
            farm_id=worker_config.farm_id,
            fleet=worker_config.fleet,
            region=worker_config.region,
            job_user=worker_config.job_user,
            job_user_group=worker_config.job_user_group,
            allow_shutdown=worker_config.allow_shutdown,
            worker_agent_install=worker_config.worker_agent_install,
            file_mappings=[
                ("/local/path/file1.txt", "/remote/path/file1.txt"),
                ("/local/path/file2.txt", "/remote/path/file2.txt"),
            ],
        )

        with patch("boto3.client"):
            worker = PosixInstanceBuildWorker(
                configuration=config_with_files,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

        with (
            patch.object(worker, "_transfer_files", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-test123"),
        ):

            # Apply the configuration
            worker.start()

            # Verify: Configuration includes file mappings
            assert worker.configuration.file_mappings is not None
            assert len(worker.configuration.file_mappings) == 2
            assert worker.configuration.file_mappings[0] == (
                "/local/path/file1.txt",
                "/remote/path/file1.txt",
            )
            assert worker.configuration.file_mappings[1] == (
                "/local/path/file2.txt",
                "/remote/path/file2.txt",
            )

            worker.stop()

    def test_property_4_configuration_with_environment_variables(
        self, worker_config, mock_worker_host
    ):
        """
        Property 4 extended: Configuration correctness with environment variables

        Test that environment variables are correctly handled in the configuration.

        **Validates: Requirements 1.3, 4.3, 5.3**
        """
        # Create configuration with environment variables
        config_with_env = DeadlineWorkerConfiguration(
            farm_id=worker_config.farm_id,
            fleet=worker_config.fleet,
            region=worker_config.region,
            job_user=worker_config.job_user,
            job_user_group=worker_config.job_user_group,
            allow_shutdown=worker_config.allow_shutdown,
            worker_agent_install=worker_config.worker_agent_install,
            worker_env_var={
                "CUSTOM_VAR_1": "value1",
                "CUSTOM_VAR_2": "value2",
                "FEATURE_FLAG": "enabled",
            },
        )

        with patch("boto3.client"):
            worker = PosixInstanceBuildWorker(
                configuration=config_with_env,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

        with (
            patch.object(worker, "_transfer_files", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-test123"),
        ):

            # Apply the configuration
            worker.start()

            # Verify: Configuration includes environment variables
            assert worker.configuration.worker_env_var is not None
            assert worker.configuration.worker_env_var["CUSTOM_VAR_1"] == "value1"
            assert worker.configuration.worker_env_var["CUSTOM_VAR_2"] == "value2"
            assert worker.configuration.worker_env_var["FEATURE_FLAG"] == "enabled"

            worker.stop()


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
            patch.object(worker, "_transfer_files", return_value=None),
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
            patch.object(worker, "_transfer_files", return_value=None),
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


class TestWorkerAgentStateCleanupProperties:
    """Property tests for worker agent state cleanup including file mappings."""

    @pytest.mark.parametrize(
        "file_mappings,expected_cleanup_files",
        [
            # Test case 1: Single file mapping
            (
                [("/tmp/source1.txt", "/home/test-user/dest1.txt")],
                ["/home/test-user/dest1.txt"],
            ),
            # Test case 2: Multiple file mappings
            (
                [
                    ("/tmp/source1.txt", "/home/test-user/dest1.txt"),
                    ("/tmp/source2.json", "/etc/config/dest2.json"),
                    ("/tmp/source3.sh", "/usr/local/bin/dest3.sh"),
                ],
                [
                    "/home/test-user/dest1.txt",
                    "/etc/config/dest2.json",
                    "/usr/local/bin/dest3.sh",
                ],
            ),
            # Test case 3: No file mappings
            (None, []),
            # Test case 4: Empty file mappings list
            ([], []),
            # Test case 5: File mappings with special characters in paths
            (
                [
                    ("/tmp/file with spaces.txt", "/home/test-user/file with spaces.txt"),
                    ("/tmp/file-with-dashes.txt", "/home/test-user/file-with-dashes.txt"),
                ],
                [
                    "/home/test-user/file with spaces.txt",
                    "/home/test-user/file-with-dashes.txt",
                ],
            ),
            # Test case 6: File mappings with nested directories
            (
                [
                    ("/tmp/src/nested/file.txt", "/home/test-user/dest/nested/file.txt"),
                    ("/tmp/another.txt", "/var/lib/another.txt"),
                ],
                [
                    "/home/test-user/dest/nested/file.txt",
                    "/var/lib/another.txt",
                ],
            ),
        ],
    )
    def test_property_9_worker_agent_state_cleanup(
        self, worker_config, mock_worker_host, file_mappings, expected_cleanup_files
    ):
        """
        Property 9: Worker agent state cleanup

        For any worker host with a running worker agent, stopping the worker agent
        should remove all worker agent state files (worker.json, configuration files, etc.)
        and clean up all staged files from file_mappings.

        **Validates: Requirements 6.2**
        """
        # Given: A worker configuration with file mappings
        config_with_files = DeadlineWorkerConfiguration(
            farm_id=worker_config.farm_id,
            fleet=worker_config.fleet,
            region=worker_config.region,
            job_user=worker_config.job_user,
            job_user_group=worker_config.job_user_group,
            allow_shutdown=worker_config.allow_shutdown,
            worker_agent_install=worker_config.worker_agent_install,
            file_mappings=file_mappings,
        )

        with patch("boto3.client"):
            worker = PosixInstanceBuildWorker(
                configuration=config_with_files,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

        # Given: A running worker agent
        mock_worker_host.is_running.return_value = True

        with (
            patch.object(worker, "_transfer_files", return_value=None),
            patch.object(worker, "get_worker_id", return_value="worker-test123"),
        ):
            worker.start()

        # When: Stopping the worker agent
        with (
            patch.object(worker, "_stop_agent_service"),
            patch.object(worker, "_cleanup_agent_state"),
            patch.object(worker, "_delete_worker"),
        ):
            worker.stop()

        # Then: All staged files should be cleaned up via worker_host.cleanup_files()
        if expected_cleanup_files:
            # Verify that cleanup_files was called on the worker_host
            mock_worker_host.cleanup_files.assert_called_once()

            # Get the file_paths argument that was passed to cleanup_files
            call_args = mock_worker_host.cleanup_files.call_args
            actual_file_paths = call_args[0][0] if call_args else []

            # Verify all expected files were passed to cleanup_files
            assert len(actual_file_paths) == len(expected_cleanup_files), (
                f"Expected {len(expected_cleanup_files)} files to be cleaned up, "
                f"but got {len(actual_file_paths)}: {actual_file_paths}"
            )
            for expected_file in expected_cleanup_files:
                assert (
                    expected_file in actual_file_paths
                ), f"Expected file {expected_file} not found in cleanup_files call: {actual_file_paths}"
        else:
            # If no file mappings, cleanup_files should not be called
            mock_worker_host.cleanup_files.assert_not_called()

        # Verify worker state is cleaned up
        assert worker.agent_state == WorkerAgentState.NOT_STARTED
        assert worker.worker_id is None

    @pytest.mark.parametrize(
        "os_type,file_paths,expected_command_pattern",
        [
            # POSIX file cleanup - single file (no special chars, no quotes needed)
            ("posix", ["/home/test-user/file.txt"], "rm -f /home/test-user/file.txt"),
            # POSIX file cleanup - multiple files
            (
                "posix",
                ["/etc/config/settings.json", "/var/log/app.log"],
                "rm -f /etc/config/settings.json /var/log/app.log",
            ),
            # Windows file cleanup - single file
            (
                "windows",
                ["C:\\Users\\test-user\\file.txt"],
                '@("C:\\Users\\test-user\\file.txt") | ForEach-Object { Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue }',
            ),
            # Windows file cleanup - multiple files
            (
                "windows",
                ["C:\\ProgramData\\config\\settings.json", "C:\\Temp\\data.txt"],
                '@("C:\\ProgramData\\config\\settings.json", "C:\\Temp\\data.txt") | ForEach-Object { Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue }',
            ),
        ],
    )
    def test_property_9_os_specific_file_cleanup_commands(
        self, os_type, file_paths, expected_command_pattern
    ):
        """
        Property 9 extended: OS-specific file cleanup commands

        File cleanup should use the appropriate OS-specific command in a single batch
        (rm -f for POSIX, Remove-Item with ForEach-Object for Windows).

        **Validates: Requirements 6.2**
        """
        # Import the appropriate host class based on OS type
        if os_type == "posix":
            from deadline_test_fixtures.deadline.worker_host import PosixEC2WorkerHost

            HostClass: Any = PosixEC2WorkerHost
        else:
            from deadline_test_fixtures.deadline.worker_host import WindowsEC2WorkerHost

            HostClass = WindowsEC2WorkerHost

        # Create a real host instance (not mocked) to test the actual command generation
        # We only need to provide the minimal required parameters for initialization
        with patch("boto3.client"):
            host = HostClass(
                subnet_id="subnet-12345",
                security_group_id="sg-12345",
                instance_profile_name="test-profile",
                bootstrap_bucket_name="test-bucket",
                s3_client=MagicMock(),
                ec2_client=MagicMock(),
                ssm_client=MagicMock(),
                instance_type="t3.micro",
                instance_shutdown_behavior="terminate",
            )

        # Test the _get_remove_files_command method directly
        actual_command = host._get_remove_files_command(file_paths)

        # Verify the command matches the expected pattern
        assert actual_command == expected_command_pattern, (
            f"Expected command:\n{expected_command_pattern}\n\n" f"But got:\n{actual_command}"
        )

        # Verify all file paths are in the command
        for file_path in file_paths:
            assert (
                file_path in actual_command
            ), f"Expected file path '{file_path}' in cleanup command: {actual_command}"


class TestErrorDiagnosticsProperties:
    """Property tests for error diagnostics and error source distinction."""

    def test_property_11_host_error_diagnostics_inclusion(self, mock_worker_host):
        """
        Property 11: Error diagnostics inclusion (Host errors)

        For any worker host operation that fails, the raised exception should contain
        diagnostic information specific to the host operation (host state, instance ID,
        operating system).

        **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
        """
        from deadline_test_fixtures.deadline.worker_host import WorkerHostError

        # Test case 1: Host error with full diagnostics
        try:
            raise WorkerHostError(
                message="Failed to start EC2 instance",
                host_os="posix",
                instance_id="i-1234567890abcdef0",
                diagnostics="Instance failed status checks\nSystem log shows kernel panic",
            )
        except WorkerHostError as e:
            # Verify error contains all diagnostic information
            assert "Failed to start EC2 instance" in str(e)
            assert "WORKER HOST ERROR" in str(e)
            assert "HOST DIAGNOSTICS" in str(e)
            assert "Operating System: posix" in str(e)
            assert "Instance ID: i-1234567890abcdef0" in str(e)
            assert "Instance failed status checks" in str(e)
            assert "System log shows kernel panic" in str(e)

            # Verify attributes are accessible
            assert e.message == "Failed to start EC2 instance"
            assert e.host_os == "posix"
            assert e.instance_id == "i-1234567890abcdef0"
            assert e.diagnostics is not None
            assert "Instance failed status checks" in e.diagnostics

        # Test case 2: Host error with minimal diagnostics
        try:
            raise WorkerHostError(
                message="Failed to send command to host",
                host_os="windows",
            )
        except WorkerHostError as e:
            # Verify error contains available diagnostic information
            assert "Failed to send command to host" in str(e)
            assert "WORKER HOST ERROR" in str(e)
            assert "Operating System: windows" in str(e)
            assert "No additional diagnostics available" in str(e)

            # Verify attributes
            assert e.message == "Failed to send command to host"
            assert e.host_os == "windows"
            assert e.instance_id is None
            assert e.diagnostics is None

    def test_property_11_agent_error_diagnostics_inclusion(self, worker_config):
        """
        Property 11: Error diagnostics inclusion (Agent errors)

        For any worker agent operation that fails, the raised exception should contain
        diagnostic information specific to the agent operation (configuration, command
        output, agent logs).

        **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
        """
        from deadline_test_fixtures.deadline.worker import WorkerAgentError
        from deadline_test_fixtures.deadline.worker_host import CommandResult

        # Test case 1: Agent error with command result
        cmd_result = CommandResult(
            exit_code=1,
            stdout="Error: Failed to configure worker agent\nConfiguration file not found",
            stderr="Permission denied",
        )

        try:
            raise WorkerAgentError(
                message="Failed to configure worker agent",
                configuration=worker_config,
                command_result=cmd_result,
                worker_id="worker-abc123",
            )
        except WorkerAgentError as e:
            # Verify error contains all diagnostic information
            assert "Failed to configure worker agent" in str(e)
            assert "WORKER AGENT ERROR" in str(e)
            assert "AGENT DIAGNOSTICS" in str(e)
            assert "Worker ID: worker-abc123" in str(e)
            assert f"Farm ID: {worker_config.farm_id}" in str(e)
            assert f"Fleet ID: {worker_config.fleet.id}" in str(e)
            assert f"Region: {worker_config.region}" in str(e)
            assert "Command Exit Code: 1" in str(e)
            assert "Failed to configure worker agent" in str(e)
            assert "Permission denied" in str(e)

            # Verify attributes are accessible
            assert e.message == "Failed to configure worker agent"
            assert e.configuration == worker_config
            assert e.command_result == cmd_result
            assert e.worker_id == "worker-abc123"

        # Test case 2: Agent error with logs
        try:
            raise WorkerAgentError(
                message="Worker agent failed to start",
                configuration=worker_config,
                logs="[ERROR] Failed to connect to Deadline Cloud service\n[ERROR] Connection timeout after 30s",
            )
        except WorkerAgentError as e:
            # Verify error contains log information
            assert "Worker agent failed to start" in str(e)
            assert "WORKER AGENT ERROR" in str(e)
            assert "Failed to connect to Deadline Cloud service" in str(e)
            assert "Connection timeout after 30s" in str(e)

            # Verify attributes
            assert e.message == "Worker agent failed to start"
            assert e.configuration == worker_config
            assert e.logs is not None
            assert "Failed to connect to Deadline Cloud service" in e.logs

        # Test case 3: Agent error with minimal diagnostics
        try:
            raise WorkerAgentError(
                message="Unknown agent error",
            )
        except WorkerAgentError as e:
            # Verify error contains minimal information
            assert "Unknown agent error" in str(e)
            assert "WORKER AGENT ERROR" in str(e)
            assert "No additional diagnostics available" in str(e)

            # Verify attributes
            assert e.message == "Unknown agent error"
            assert e.configuration is None
            assert e.command_result is None
            assert e.logs is None
            assert e.worker_id is None

    @pytest.mark.parametrize(
        "error_type,error_params,expected_diagnostics",
        [
            # Host error with various diagnostic combinations
            (
                "host",
                {
                    "message": "Instance launch failed",
                    "host_os": "posix",
                    "instance_id": "i-abc123",
                    "diagnostics": "AMI not found in region",
                },
                [
                    "WORKER HOST ERROR",
                    "Operating System: posix",
                    "Instance ID: i-abc123",
                    "AMI not found",
                ],
            ),
            (
                "host",
                {
                    "message": "SSM command timeout",
                    "host_os": "windows",
                    "diagnostics": "SSM agent not responding",
                },
                ["WORKER HOST ERROR", "Operating System: windows", "SSM agent not responding"],
            ),
            # Agent error with various diagnostic combinations
            (
                "agent",
                {
                    "message": "Agent installation failed",
                    "worker_id": "worker-xyz789",
                    "logs": "pip install failed\nPackage not found",
                },
                [
                    "WORKER AGENT ERROR",
                    "Worker ID: worker-xyz789",
                    "pip install failed",
                    "Package not found",
                ],
            ),
            (
                "agent",
                {
                    "message": "Agent configuration failed",
                },
                [
                    "WORKER AGENT ERROR",
                    "Agent configuration failed",
                    "No additional diagnostics available",
                ],
            ),
        ],
    )
    def test_property_11_various_error_diagnostics(
        self, error_type, error_params, expected_diagnostics
    ):
        """
        Property 11 extended: Error diagnostics with various error scenarios

        Test that error diagnostics are included for various types of failures.

        **Validates: Requirements 6.1, 6.2, 6.3, 6.4**
        """
        if error_type == "host":
            from deadline_test_fixtures.deadline.worker_host import WorkerHostError

            try:
                raise WorkerHostError(**error_params)
            except WorkerHostError as e:
                error_str = str(e)
                for expected in expected_diagnostics:
                    assert (
                        expected in error_str
                    ), f"Expected diagnostic '{expected}' not found in error message:\n{error_str}"
        else:  # agent
            from deadline_test_fixtures.deadline.worker import WorkerAgentError

            try:
                raise WorkerAgentError(**error_params)
            except WorkerAgentError as e:
                error_str = str(e)
                for expected in expected_diagnostics:
                    assert (
                        expected in error_str
                    ), f"Expected diagnostic '{expected}' not found in error message:\n{error_str}"

    def test_property_12_error_source_distinction(self, worker_config):
        """
        Property 12: Error source distinction

        For any failure, the error message should clearly indicate whether the failure
        originated from the worker host layer or the worker agent layer.

        **Validates: Requirements 6.5**
        """
        from deadline_test_fixtures.deadline.worker import WorkerAgentError
        from deadline_test_fixtures.deadline.worker_host import WorkerHostError

        # Test case 1: Host error clearly indicates host-level failure
        try:
            raise WorkerHostError(
                message="EC2 instance failed to start",
                host_os="posix",
                instance_id="i-123456",
            )
        except WorkerHostError as e:
            error_str = str(e)
            # Verify clear indication of host-level error
            assert "WORKER HOST ERROR" in error_str
            assert "HOST DIAGNOSTICS" in error_str
            # Verify it doesn't contain agent-related terminology
            assert "WORKER AGENT ERROR" not in error_str
            assert "AGENT DIAGNOSTICS" not in error_str

        # Test case 2: Agent error clearly indicates agent-level failure
        try:
            raise WorkerAgentError(
                message="Worker agent failed to configure",
                configuration=worker_config,
                worker_id="worker-abc123",
            )
        except WorkerAgentError as e:
            error_str = str(e)
            # Verify clear indication of agent-level error
            assert "WORKER AGENT ERROR" in error_str
            assert "AGENT DIAGNOSTICS" in error_str
            # Verify it doesn't contain host-related terminology
            assert "WORKER HOST ERROR" not in error_str
            assert "HOST DIAGNOSTICS" not in error_str

        # Test case 3: Different error types are distinguishable
        host_error_msg = None
        agent_error_msg = None

        try:
            raise WorkerHostError(message="Host failure", host_os="windows")
        except WorkerHostError as e:
            host_error_msg = str(e)

        try:
            raise WorkerAgentError(message="Agent failure")
        except WorkerAgentError as e:
            agent_error_msg = str(e)

        # Verify the error messages are clearly different
        assert host_error_msg != agent_error_msg
        assert "WORKER HOST ERROR" in host_error_msg
        assert "WORKER AGENT ERROR" in agent_error_msg
        assert "HOST DIAGNOSTICS" in host_error_msg
        assert "AGENT DIAGNOSTICS" in agent_error_msg

    @pytest.mark.parametrize(
        "error_scenario,error_class,expected_markers",
        [
            # Host-level errors
            (
                "EC2 instance launch failure",
                "WorkerHostError",
                ["WORKER HOST ERROR", "HOST DIAGNOSTICS"],
            ),
            (
                "SSM command execution failure",
                "WorkerHostError",
                ["WORKER HOST ERROR", "HOST DIAGNOSTICS"],
            ),
            (
                "Instance status check failure",
                "WorkerHostError",
                ["WORKER HOST ERROR", "HOST DIAGNOSTICS"],
            ),
            # Agent-level errors
            (
                "Worker agent installation failure",
                "WorkerAgentError",
                ["WORKER AGENT ERROR", "AGENT DIAGNOSTICS"],
            ),
            (
                "Worker agent configuration failure",
                "WorkerAgentError",
                ["WORKER AGENT ERROR", "AGENT DIAGNOSTICS"],
            ),
            (
                "Worker agent service start failure",
                "WorkerAgentError",
                ["WORKER AGENT ERROR", "AGENT DIAGNOSTICS"],
            ),
        ],
    )
    def test_property_12_various_error_sources(self, error_scenario, error_class, expected_markers):
        """
        Property 12 extended: Error source distinction for various failure scenarios

        Test that error source is clearly distinguished for various types of failures.

        **Validates: Requirements 6.5**
        """
        if error_class == "WorkerHostError":
            from deadline_test_fixtures.deadline.worker_host import WorkerHostError

            try:
                raise WorkerHostError(
                    message=error_scenario,
                    host_os="posix",
                )
            except WorkerHostError as e:
                error_str = str(e)
                for marker in expected_markers:
                    assert (
                        marker in error_str
                    ), f"Expected marker '{marker}' not found in {error_class} for scenario '{error_scenario}'"
                # Verify no agent markers
                assert "WORKER AGENT ERROR" not in error_str
                assert "AGENT DIAGNOSTICS" not in error_str
        else:  # WorkerAgentError
            from deadline_test_fixtures.deadline.worker import WorkerAgentError

            try:
                raise WorkerAgentError(
                    message=error_scenario,
                )
            except WorkerAgentError as e:
                error_str = str(e)
                for marker in expected_markers:
                    assert (
                        marker in error_str
                    ), f"Expected marker '{marker}' not found in {error_class} for scenario '{error_scenario}'"
                # Verify no host markers
                assert "WORKER HOST ERROR" not in error_str
                assert "HOST DIAGNOSTICS" not in error_str

    def test_property_12_command_failure_error_source(self, worker_config, mock_worker_host):
        """
        Property 12 extended: Command failures should indicate the appropriate error source

        When a command fails, the error should indicate whether it was a host-level
        command failure or an agent-level command failure.

        **Validates: Requirements 6.5**
        """
        from deadline_test_fixtures.deadline.worker import WorkerAgentError
        from deadline_test_fixtures.deadline.worker_host import CommandResult, WorkerHostError

        # Test case 1: Host-level command failure (e.g., SSM command to host)
        try:
            raise WorkerHostError(
                message="Failed to send command to host",
                host_os="posix",
                instance_id="i-123456",
                diagnostics="SSM command timed out after 30 seconds",
            )
        except WorkerHostError as e:
            error_str = str(e)
            assert "WORKER HOST ERROR" in error_str
            assert "Failed to send command to host" in error_str
            assert "SSM command timed out" in error_str

        # Test case 2: Agent-level command failure (e.g., agent configuration command)
        cmd_result = CommandResult(
            exit_code=1,
            stdout="",
            stderr="install-deadline-worker: command not found",
        )

        try:
            raise WorkerAgentError(
                message="Failed to install worker agent",
                configuration=worker_config,
                command_result=cmd_result,
            )
        except WorkerAgentError as e:
            error_str = str(e)
            assert "WORKER AGENT ERROR" in error_str
            assert "Failed to install worker agent" in error_str
            assert "install-deadline-worker: command not found" in error_str
            assert "Command Exit Code: 1" in error_str


class TestConfigurationAcceptanceProperties:
    """Property tests for configuration acceptance."""

    def test_property_20_configuration_acceptance(self, worker_config, mock_worker_host):
        """
        Property 20: Configuration acceptance

        For any worker agent setup operation, the operation should accept a
        DeadlineWorkerConfiguration object containing all necessary configuration parameters.

        **Validates: Requirements 5.1**
        """
        with patch("boto3.client"):
            # Given: A DeadlineWorkerConfiguration object with all necessary parameters
            # When: Creating a worker with the configuration
            worker = PosixInstanceBuildWorker(
                configuration=worker_config,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

            # Then: Worker should accept the configuration
            assert worker.configuration == worker_config
            assert worker.configuration.farm_id == "farm-123"
            assert worker.configuration.fleet.id == "fleet_123"
            assert worker.configuration.region == "us-west-2"
            assert worker.configuration.job_user == "test-user"
            assert worker.configuration.job_user_group == "test-group"
            assert worker.configuration.allow_shutdown is False

            # When: Starting the worker agent with the configuration
            with (
                patch.object(worker, "_transfer_files", return_value=None),
                patch.object(worker, "get_worker_id", return_value="worker-test123"),
            ):
                worker.start()

                # Then: Worker agent should be running with the configuration
                assert worker.agent_state == WorkerAgentState.RUNNING
                assert worker.worker_id == "worker-test123"

                # Verify the configuration was used during agent setup
                # The configure_worker_command should be called with the configuration
                configure_cmd = worker.configure_worker_command(config=worker_config)
                assert "farm-123" in configure_cmd
                assert "fleet_123" in configure_cmd
                assert "us-west-2" in configure_cmd

                worker.stop()

    @pytest.mark.parametrize(
        "config_params",
        [
            # Minimal configuration
            {
                "farm_id": "farm-minimal",
                "fleet_id": "fleet-minimal",
                "region": "us-west-2",
                "job_user": "job-user",
                "job_user_group": "job-group",
                "allow_shutdown": False,
            },
            # Configuration with file mappings
            {
                "farm_id": "farm-files",
                "fleet_id": "fleet-files",
                "region": "us-east-1",
                "job_user": "job-user",
                "job_user_group": "job-group",
                "allow_shutdown": True,
                "file_mappings": [
                    ("/local/file1.txt", "/remote/file1.txt"),
                    ("/local/file2.json", "/remote/file2.json"),
                ],
            },
            # Configuration with environment variables
            {
                "farm_id": "farm-env",
                "fleet_id": "fleet-env",
                "region": "eu-west-1",
                "job_user": "custom-user",
                "job_user_group": "custom-group",
                "allow_shutdown": True,
                "worker_env_var": {
                    "CUSTOM_VAR": "value",
                    "FEATURE_FLAG": "enabled",
                },
            },
            # Configuration with pre-install commands
            {
                "farm_id": "farm-preinstall",
                "fleet_id": "fleet-preinstall",
                "region": "ap-southeast-1",
                "job_user": "job-user",
                "job_user_group": "job-group",
                "allow_shutdown": False,
                "pre_install_commands": [
                    "apt-get update",
                    "apt-get install -y python3-pip",
                ],
            },
            # Configuration with session root directory
            {
                "farm_id": "farm-session",
                "fleet_id": "fleet-session",
                "region": "us-west-1",
                "job_user": "job-user",
                "job_user_group": "job-group",
                "allow_shutdown": True,
                "session_root_dir": "/mnt/sessions",
            },
            # Configuration with all optional parameters
            {
                "farm_id": "farm-full",
                "fleet_id": "fleet-full",
                "region": "us-east-2",
                "job_user": "full-user",
                "job_user_group": "full-group",
                "allow_shutdown": True,
                "file_mappings": [("/src/config.yaml", "/etc/config.yaml")],
                "pre_install_commands": ["echo 'Setup starting'"],
                "worker_env_var": {"ENV_VAR": "value"},
                "session_root_dir": "/custom/sessions",
                "start_service": True,
                "no_install_service": False,
            },
        ],
    )
    def test_property_20_various_configurations(self, config_params, mock_worker_host):
        """
        Property 20 extended: Configuration acceptance with various parameter combinations

        Test that worker agent setup accepts DeadlineWorkerConfiguration objects with
        various combinations of required and optional parameters.

        **Validates: Requirements 5.1**
        """
        # Create configuration with specific parameters
        config = DeadlineWorkerConfiguration(
            farm_id=config_params["farm_id"],
            fleet=Fleet(
                id=config_params["fleet_id"],
                farm=Farm(id=config_params["farm_id"]),
            ),
            region=config_params["region"],
            job_user=config_params["job_user"],
            job_user_group=config_params["job_user_group"],
            allow_shutdown=config_params["allow_shutdown"],
            worker_agent_install=PipInstall(
                requirement_specifiers=["deadline-cloud-worker-agent"],
                codeartifact=CodeArtifactRepositoryInfo(
                    region=config_params["region"],
                    domain="test-domain",
                    domain_owner="123456789123",
                    repository="test-repository",
                ),
            ),
            file_mappings=config_params.get("file_mappings"),
            pre_install_commands=config_params.get("pre_install_commands"),
            worker_env_var=config_params.get("worker_env_var"),
            session_root_dir=config_params.get("session_root_dir"),
            start_service=config_params.get("start_service", True),
            no_install_service=config_params.get("no_install_service", False),
        )

        with patch("boto3.client"):
            # When: Creating a worker with the configuration
            worker = PosixInstanceBuildWorker(
                configuration=config,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

            # Then: Worker should accept the configuration
            assert worker.configuration == config
            assert worker.configuration.farm_id == config_params["farm_id"]
            assert worker.configuration.fleet.id == config_params["fleet_id"]
            assert worker.configuration.region == config_params["region"]
            assert worker.configuration.job_user == config_params["job_user"]
            assert worker.configuration.job_user_group == config_params["job_user_group"]
            assert worker.configuration.allow_shutdown == config_params["allow_shutdown"]

            # Verify optional parameters
            if "file_mappings" in config_params:
                assert worker.configuration.file_mappings == config_params["file_mappings"]
            if "pre_install_commands" in config_params:
                assert (
                    worker.configuration.pre_install_commands
                    == config_params["pre_install_commands"]
                )
            if "worker_env_var" in config_params:
                assert worker.configuration.worker_env_var == config_params["worker_env_var"]
            if "session_root_dir" in config_params:
                assert worker.configuration.session_root_dir == config_params["session_root_dir"]

            # When: Starting the worker agent
            with (
                patch.object(worker, "_transfer_files", return_value=None),
                patch.object(worker, "get_worker_id", return_value="worker-test123"),
            ):
                worker.start()

                # Then: Worker agent should be running with the configuration
                assert worker.agent_state == WorkerAgentState.RUNNING
                assert worker.worker_id == "worker-test123"

                worker.stop()

    def test_property_20_configuration_with_host_reuse(self, mock_worker_host):
        """
        Property 20 extended: Configuration acceptance supports host reuse scenarios

        Test that multiple different configurations can be applied sequentially to the
        same host, demonstrating that configuration acceptance supports host reuse.

        **Validates: Requirements 5.1, 5.3, 5.5**
        """
        # Create three different configurations
        config_a = DeadlineWorkerConfiguration(
            farm_id="farm-a",
            fleet=Fleet(id="fleet-a", farm=Farm(id="farm-a")),
            region="us-west-2",
            job_user="user-a",
            job_user_group="group-a",
            allow_shutdown=False,
            worker_agent_install=PipInstall(
                requirement_specifiers=["deadline-cloud-worker-agent==1.0.0"],
                codeartifact=CodeArtifactRepositoryInfo(
                    region="us-west-2",
                    domain="domain-a",
                    domain_owner="123456789123",
                    repository="repo-a",
                ),
            ),
            file_mappings=[("/local/a.txt", "/remote/a.txt")],
        )

        config_b = DeadlineWorkerConfiguration(
            farm_id="farm-b",
            fleet=Fleet(id="fleet-b", farm=Farm(id="farm-b")),
            region="us-east-1",
            job_user="user-b",
            job_user_group="group-b",
            allow_shutdown=True,
            worker_agent_install=PipInstall(
                requirement_specifiers=["deadline-cloud-worker-agent==2.0.0"],
                codeartifact=CodeArtifactRepositoryInfo(
                    region="us-east-1",
                    domain="domain-b",
                    domain_owner="987654321987",
                    repository="repo-b",
                ),
            ),
            worker_env_var={"ENV_B": "value_b"},
        )

        config_c = DeadlineWorkerConfiguration(
            farm_id="farm-c",
            fleet=Fleet(id="fleet-c", farm=Farm(id="farm-c")),
            region="eu-west-1",
            job_user="user-c",
            job_user_group="group-c",
            allow_shutdown=True,
            worker_agent_install=PipInstall(
                requirement_specifiers=["deadline-cloud-worker-agent==3.0.0"],
                codeartifact=CodeArtifactRepositoryInfo(
                    region="eu-west-1",
                    domain="domain-c",
                    domain_owner="111222333444",
                    repository="repo-c",
                ),
            ),
            session_root_dir="/custom/sessions",
        )

        with patch("boto3.client"):
            # Create three workers with different configurations (same host)
            worker_a = PosixInstanceBuildWorker(
                configuration=config_a,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )
            worker_b = PosixInstanceBuildWorker(
                configuration=config_b,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )
            worker_c = PosixInstanceBuildWorker(
                configuration=config_c,
                worker_host=mock_worker_host,
                deadline_client=MagicMock(),
            )

        with (
            patch.object(worker_a, "_transfer_files", return_value=None),
            patch.object(worker_a, "get_worker_id", return_value="worker-aaa"),
            patch.object(worker_b, "_transfer_files", return_value=None),
            patch.object(worker_b, "get_worker_id", return_value="worker-bbb"),
            patch.object(worker_c, "_transfer_files", return_value=None),
            patch.object(worker_c, "get_worker_id", return_value="worker-ccc"),
        ):
            # Apply configuration A
            worker_a.start()
            assert worker_a.configuration == config_a
            assert worker_a.configuration.farm_id == "farm-a"
            assert worker_a.configuration.file_mappings == [("/local/a.txt", "/remote/a.txt")]
            worker_a.stop()

            # Apply configuration B to the same host
            worker_b.start()
            assert worker_b.configuration == config_b
            assert worker_b.configuration.farm_id == "farm-b"
            assert worker_b.configuration.worker_env_var == {"ENV_B": "value_b"}
            worker_b.stop()

            # Apply configuration C to the same host
            worker_c.start()
            assert worker_c.configuration == config_c
            assert worker_c.configuration.farm_id == "farm-c"
            assert worker_c.configuration.session_root_dir == "/custom/sessions"
            worker_c.stop()

    @pytest.mark.parametrize(
        "os_type,worker_class",
        [
            ("posix", "PosixInstanceBuildWorker"),
            ("windows", "WindowsInstanceBuildWorker"),
        ],
    )
    def test_property_20_configuration_acceptance_cross_platform(
        self, os_type, worker_class, worker_config
    ):
        """
        Property 20 extended: Configuration acceptance works across platforms

        Test that DeadlineWorkerConfiguration is accepted by both Windows and POSIX workers.

        **Validates: Requirements 5.1**
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

        # Add all the required attributes
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
            # When: Creating a worker with the configuration
            worker = WorkerClass(
                configuration=worker_config,
                worker_host=mock_host,
                deadline_client=MagicMock(),
            )

            # Then: Worker should accept the configuration regardless of OS
            assert worker.configuration == worker_config
            assert worker.configuration.farm_id == "farm-123"
            assert worker.configuration.fleet.id == "fleet_123"

            # When: Starting the worker agent
            with (
                patch.object(worker, "_transfer_files", return_value=None),
                patch.object(worker, "get_worker_id", return_value="worker-test123"),
            ):
                worker.start()

                # Then: Worker agent should be running with the configuration
                assert worker.agent_state == WorkerAgentState.RUNNING
                assert worker.worker_id == "worker-test123"

                worker.stop()
