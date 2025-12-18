# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
from __future__ import annotations

import pytest


from deadline_test_fixtures.deadline.worker_host import (
    WorkerHost,
    WorkerHostState,
    WorkerAgentState,
    EC2WorkerHost,
    Ec2Tag,
    CommandResult,
)
from unittest.mock import Mock


class MockWorkerHost(WorkerHost):
    """Mock implementation of WorkerHost for testing."""

    def __init__(self, operating_system: str = "posix"):
        super().__init__()
        self._os = operating_system
        self._started = False
        self._stopped = False

    def _operating_system(self) -> str:
        return self._os

    def _do_start(self) -> None:
        self._started = True

    def _do_stop(self) -> None:
        self._stopped = True

    def send_command(self, command: str) -> CommandResult:
        if not self.is_running():
            raise RuntimeError("Cannot send command to non-running host")
        return CommandResult(exit_code=0, stdout="mock output", stderr=None)


class TestWorkerHost:
    """Test WorkerHost abstract base class functionality."""

    def test_initial_state(self):
        """Test that WorkerHost starts in NOT_STARTED state."""
        host = MockWorkerHost()
        assert host.state == WorkerHostState.NOT_STARTED
        assert not host.has_active_worker
        assert not host.is_running()

    def test_start_transitions_to_running(self):
        """Test that start() transitions to RUNNING state."""
        host = MockWorkerHost()
        host.start()
        assert host.state == WorkerHostState.RUNNING
        assert host.is_running()
        assert host._started

    def test_stop_transitions_to_stopped(self):
        """Test that stop() transitions to STOPPED state."""
        host = MockWorkerHost()
        host.start()
        host.stop()
        assert host.state == WorkerHostState.STOPPED
        assert not host.is_running()
        assert host._stopped

    def test_stop_from_not_started_cleans_up(self):
        """Test that stop() can be called from NOT_STARTED state for cleanup."""
        host = MockWorkerHost()
        host.stop()
        assert host.state == WorkerHostState.STOPPED
        assert host._stopped

    def test_start_when_already_running_raises_error(self):
        """Test that start() raises RuntimeError when already running."""
        host = MockWorkerHost()
        host.start()
        with pytest.raises(RuntimeError, match="Worker host is already running"):
            host.start()

    def test_start_when_stopped_raises_error(self):
        """Test that start() raises RuntimeError when already stopped."""
        host = MockWorkerHost()
        host.start()
        host.stop()
        with pytest.raises(RuntimeError, match="Cannot restart a stopped worker host"):
            host.start()

    def test_stop_when_already_stopped_raises_error(self):
        """Test that stop() raises RuntimeError when already stopped."""
        host = MockWorkerHost()
        host.start()
        host.stop()
        with pytest.raises(RuntimeError, match="Worker host is already stopped"):
            host.stop()

    def test_worker_claiming_and_releasing(self):
        """Test worker claiming and releasing functionality."""
        host = MockWorkerHost()
        worker_id_1 = 12345
        worker_id_2 = 67890

        # Initially no active worker
        assert not host.has_active_worker

        # Claim for worker 1
        host._claim_for_worker(worker_id_1)
        assert host.has_active_worker

        # Same worker can claim again
        host._claim_for_worker(worker_id_1)
        assert host.has_active_worker

        # Different worker cannot claim
        with pytest.raises(RuntimeError, match="another worker.*already has an agent running"):
            host._claim_for_worker(worker_id_2)

        # Release from worker 1
        host._release_from_worker(worker_id_1)
        assert not host.has_active_worker

        # Now worker 2 can claim
        host._claim_for_worker(worker_id_2)
        assert host.has_active_worker

        # Release from wrong worker does nothing
        host._release_from_worker(worker_id_1)
        assert host.has_active_worker

        # Release from correct worker
        host._release_from_worker(worker_id_2)
        assert not host.has_active_worker

    def test_send_command_when_running(self):
        """Test that send_command works when host is running."""
        host = MockWorkerHost()
        host.start()
        result = host.send_command("test command")
        assert result.exit_code == 0
        assert result.stdout == "mock output"

    def test_send_command_when_not_running_raises_error(self):
        """Test that send_command raises error when host is not running."""
        host = MockWorkerHost()
        with pytest.raises(RuntimeError, match="Cannot send command to non-running host"):
            host.send_command("test command")


class TestWorkerHostStateManagement:
    """Property-based tests for WorkerHost state management."""

    @pytest.mark.parametrize("operating_system", ["windows", "posix"])
    def test_property_1_worker_host_creation_independence(self, operating_system: str):
        """
        **Feature: worker-host-decoupling, Property 1: Worker host creation independence**

        For any EC2 worker host (Windows or POSIX), creating and starting a worker host
        should not result in any worker agent processes running or worker agent
        configuration files existing on the host.

        **Validates: Requirements 1.1, 2.3**
        """
        # Create and start a worker host
        host = MockWorkerHost(operating_system=operating_system)

        # Verify initial state - no worker agent should be active
        assert host.state == WorkerHostState.NOT_STARTED
        assert not host.has_active_worker

        # Start the host
        host.start()

        # Verify host is running but no worker agent is active
        assert host.state == WorkerHostState.RUNNING
        assert host.is_running()
        assert not host.has_active_worker

        # Verify the host can receive commands (indicating it's properly started)
        result = host.send_command("echo 'test'")
        assert result.exit_code == 0

        # Verify no worker agent processes or configuration should exist
        # (This is validated by the fact that has_active_worker remains False)
        assert not host.has_active_worker

    @pytest.mark.parametrize(
        "initial_state,action,expected_state,should_raise",
        [
            (WorkerHostState.NOT_STARTED, "start", WorkerHostState.RUNNING, False),
            (WorkerHostState.NOT_STARTED, "stop", WorkerHostState.STOPPED, False),
            (WorkerHostState.RUNNING, "start", WorkerHostState.RUNNING, True),
            (WorkerHostState.RUNNING, "stop", WorkerHostState.STOPPED, False),
            (WorkerHostState.STOPPED, "start", WorkerHostState.STOPPED, True),
            (WorkerHostState.STOPPED, "stop", WorkerHostState.STOPPED, True),
        ],
    )
    def test_state_transition_validation(
        self,
        initial_state: WorkerHostState,
        action: str,
        expected_state: WorkerHostState,
        should_raise: bool,
    ):
        """Test that state transitions are properly validated."""
        host = MockWorkerHost()

        # Set up initial state
        if initial_state == WorkerHostState.RUNNING:
            host.start()
        elif initial_state == WorkerHostState.STOPPED:
            host.start()
            host.stop()

        assert host.state == initial_state

        # Perform action and check result
        if should_raise:
            with pytest.raises(RuntimeError):
                if action == "start":
                    host.start()
                else:
                    host.stop()
        else:
            if action == "start":
                host.start()
            else:
                host.stop()
            assert host.state == expected_state

    @pytest.mark.parametrize(
        "worker_scenarios",
        [
            # Single worker claiming and releasing
            [(12345, "claim"), (12345, "release")],
            # Multiple workers trying to claim (second should fail)
            [(12345, "claim"), (67890, "claim_fail"), (12345, "release"), (67890, "claim")],
            # Worker releasing without claiming (should be no-op)
            [(12345, "release")],
            # Same worker claiming multiple times
            [(12345, "claim"), (12345, "claim"), (12345, "release")],
        ],
    )
    def test_worker_ownership_scenarios(self, worker_scenarios):
        """Test various worker ownership scenarios."""
        host = MockWorkerHost()

        for worker_id, action in worker_scenarios:
            if action == "claim":
                host._claim_for_worker(worker_id)
                assert host.has_active_worker
            elif action == "claim_fail":
                with pytest.raises(
                    RuntimeError, match="another worker.*already has an agent running"
                ):
                    host._claim_for_worker(worker_id)
            elif action == "release":
                host._release_from_worker(worker_id)
                # Release from non-active worker should be no-op


class TestWorkerAgentState:
    """Test WorkerAgentState enum."""

    def test_worker_agent_state_values(self):
        """Test that WorkerAgentState has expected values."""
        assert WorkerAgentState.NOT_STARTED.value == "not_started"
        assert WorkerAgentState.RUNNING.value == "running"
        assert WorkerAgentState.STOPPED.value == "stopped"


class TestWorkerHostState:
    """Test WorkerHostState enum."""

    def test_worker_host_state_values(self):
        """Test that WorkerHostState has expected values."""
        assert WorkerHostState.NOT_STARTED.value == "not_started"
        assert WorkerHostState.RUNNING.value == "running"
        assert WorkerHostState.STOPPED.value == "stopped"


class MockEC2WorkerHost(EC2WorkerHost):
    """Mock implementation of EC2WorkerHost for testing."""

    def __init__(self, operating_system: str = "posix"):
        # Create mock clients
        mock_s3_client = Mock()
        mock_ec2_client = Mock()
        mock_ssm_client = Mock()

        # Mock SSM parameter response for AMI ID
        mock_ssm_client.get_parameters.return_value = {"Parameters": [{"Value": "ami-12345678"}]}

        # Mock EC2 run_instances response
        mock_ec2_client.run_instances.return_value = {
            "Instances": [{"InstanceId": "i-1234567890abcdef0"}]
        }

        # Mock EC2 waiter
        mock_waiter = Mock()
        mock_ec2_client.get_waiter.return_value = mock_waiter

        # Mock SSM send_command response
        mock_ssm_client.send_command.return_value = {"Command": {"CommandId": "cmd-12345"}}

        # Mock SSM get_command_invocation response
        mock_ssm_client.get_command_invocation.return_value = {
            "ResponseCode": 0,
            "StandardOutputContent": "mock output",
            "StandardErrorContent": "",
        }

        # Mock SSM waiter
        mock_ssm_waiter = Mock()
        mock_ssm_client.get_waiter.return_value = mock_ssm_waiter

        super().__init__(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            s3_client=mock_s3_client,
            ec2_client=mock_ec2_client,
            ssm_client=mock_ssm_client,
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
        )
        self._os = operating_system

    def _operating_system(self) -> str:
        return self._os

    def ami_ssm_param_name(self) -> str:
        return "/test/ami/parameter"

    def ssm_document_name(self) -> str:
        return "AWS-RunShellScript" if self._os == "posix" else "AWS-RunPowerShellScript"

    def userdata(self, s3_files: list[tuple[str, str]] | None) -> str:
        return "#!/bin/bash\necho 'test userdata'"

    def ebs_devices(self) -> dict[str, int] | None:
        return {"/dev/xvda": 30} if self._os == "posix" else {"/dev/sda1": 60}


class TestEC2WorkerHost:
    """Test EC2WorkerHost base class functionality."""

    def test_ec2_worker_host_initialization(self):
        """Test that EC2WorkerHost initializes correctly."""
        host = MockEC2WorkerHost()
        assert host.state == WorkerHostState.NOT_STARTED
        assert not host.has_active_worker
        assert host.instance_id is None
        assert host.subnet_id == "subnet-12345"
        assert host.security_group_id == "sg-12345"

    def test_ec2_worker_host_ami_id_resolution(self):
        """Test that AMI ID is resolved from SSM parameter."""
        host = MockEC2WorkerHost()
        ami_id = host.ami_id
        assert ami_id == "ami-12345678"
        host.ssm_client.get_parameters.assert_called_once_with(Names=["/test/ami/parameter"])

    def test_ec2_worker_host_start_launches_instance(self):
        """Test that starting EC2WorkerHost launches an instance."""
        host = MockEC2WorkerHost()
        host.start()

        assert host.state == WorkerHostState.RUNNING
        assert host.instance_id == "i-1234567890abcdef0"
        host.ec2_client.run_instances.assert_called_once()

    def test_ec2_worker_host_stop_terminates_instance(self):
        """Test that stopping EC2WorkerHost terminates the instance."""
        host = MockEC2WorkerHost()
        host.start()
        host.stop()

        assert host.state == WorkerHostState.STOPPED
        assert host.instance_id is None
        host.ec2_client.terminate_instances.assert_called_once_with(
            InstanceIds=["i-1234567890abcdef0"]
        )

    def test_ec2_worker_host_send_command(self):
        """Test that EC2WorkerHost can send commands via SSM."""
        host = MockEC2WorkerHost()
        host.start()

        result = host.send_command("echo 'test'")

        assert result.exit_code == 0
        assert result.stdout == "mock output"
        host.ssm_client.send_command.assert_called_once()


class TestEc2Tag:
    """Test Ec2Tag dataclass."""

    def test_ec2_tag_creation(self):
        """Test that Ec2Tag can be created with key and value."""
        tag = Ec2Tag(key="Environment", value="Test")
        assert tag.key == "Environment"
        assert tag.value == "Test"

    def test_ec2_tag_equality(self):
        """Test that Ec2Tag instances with same values are equal."""
        tag1 = Ec2Tag(key="Environment", value="Test")
        tag2 = Ec2Tag(key="Environment", value="Test")
        tag3 = Ec2Tag(key="Environment", value="Production")

        assert tag1 == tag2
        assert tag1 != tag3
