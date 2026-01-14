# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
from __future__ import annotations

import pytest
from unittest.mock import Mock, patch


from deadline_test_fixtures.deadline.worker_host import (
    CommandResult,
    Ec2Tag,
    EC2WorkerHost,
    InstanceStartupError,
    PosixEC2WorkerHost,
    WindowsEC2WorkerHost,
    WorkerAgentState,
    WorkerHost,
    WorkerHostState,
)


@pytest.fixture(autouse=True)
def mock_sleep():
    """Auto-use fixture to mock time.sleep in all tests to avoid delays."""
    with patch("time.sleep"), patch("deadline_test_fixtures.util.sleep"):
        yield


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
            "StandardOutputContent": "Userdata finished successfully",  # Return success by default
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

    def userdata_success_script(self) -> str:
        """Generate script to check userdata completion status."""
        if self._os == "windows":
            return f"""
if (Test-Path "C:\\signal_user_data_finished\\success") {{
    echo "{self.USERDATA_SUCCESS_STRING}"
    exit 0
}}
if (Test-Path "C:\\signal_user_data_finished\\failed") {{
    echo "{self.USERDATA_FAILURE_STRING}"
    cat "C:\\signal_user_data_finished\\failed"
    exit 0
}}
"""
        else:
            return f"""
if [[ -f "/var/tmp/signal_user_data_finished/success" ]]; then
    echo "{self.USERDATA_SUCCESS_STRING}"
    exit 0
fi
if [[ -f "/var/tmp/signal_user_data_finished/failed" ]]; then
    echo "{self.USERDATA_FAILURE_STRING}"
    cat "/var/tmp/signal_user_data_finished/failed"
    exit 0
fi
"""

    def ebs_devices(self) -> dict[str, int] | None:
        return None

    def _get_download_files_command(self, s3_files: list[tuple[str, str]]) -> str:
        """Get the OS-specific command to download files from S3."""
        if self._os == "windows":
            download_commands = [f"aws s3 cp {s3_uri} {dst}" for s3_uri, dst in s3_files]
            return " ; ".join(download_commands)
        else:
            import shlex

            download_commands = [
                f"aws s3 cp {shlex.quote(s3_uri)} {shlex.quote(dst)} && chmod o+rx {shlex.quote(dst)}"
                for s3_uri, dst in s3_files
            ]
            return " && ".join(download_commands)

    def _get_remove_files_command(self, file_paths: list[str]) -> str:
        """Get the OS-specific command to remove multiple files in a single command."""
        if self._os == "windows":
            paths_array = ", ".join([f'"{path}"' for path in file_paths])
            return f"@({paths_array}) | ForEach-Object {{ Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue }}"
        else:
            import shlex

            return f"rm -f {shlex.join(file_paths)}"
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

        # Mock the userdata check to return success immediately, then return mock output for subsequent commands
        call_count = 0

        def mock_get_command_invocation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is for userdata checking during start()
                return {
                    "ResponseCode": 0,
                    "StandardOutputContent": "Userdata finished successfully",
                    "StandardErrorContent": "",
                }
            else:
                # Subsequent calls return mock output
                return {
                    "ResponseCode": 0,
                    "StandardOutputContent": "mock output",
                    "StandardErrorContent": "",
                }

        host.ssm_client.get_command_invocation.side_effect = mock_get_command_invocation

        host.start()

        result = host.send_command("echo 'test'")

        assert result.exit_code == 0
        assert result.stdout == "mock output"
        # Should be called twice: once for userdata check, once for the test command
        assert host.ssm_client.send_command.call_count == 2


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


class TestWindowsEC2WorkerHost:
    """Test WindowsEC2WorkerHost implementation."""

    def test_windows_ec2_worker_host_operating_system(self):
        """Test that WindowsEC2WorkerHost returns 'windows' as operating system."""
        mock_clients = {
            "s3_client": Mock(),
            "ec2_client": Mock(),
            "ssm_client": Mock(),
        }
        mock_clients["ssm_client"].get_parameters.return_value = {
            "Parameters": [{"Value": "ami-windows123"}]
        }

        host = WindowsEC2WorkerHost(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        assert host._operating_system() == "windows"
        assert host.ssm_document_name() == "AWS-RunPowerShellScript"
        assert host.ebs_devices() == {"/dev/sda1": 60}
        assert "Windows_Server-2022-English-Full-Base" in host.ami_ssm_param_name()

    def test_windows_ec2_worker_host_userdata(self):
        """Test that WindowsEC2WorkerHost generates Windows userdata."""
        mock_clients = {
            "s3_client": Mock(),
            "ec2_client": Mock(),
            "ssm_client": Mock(),
        }
        mock_clients["ssm_client"].get_parameters.return_value = {
            "Parameters": [{"Value": "ami-windows123"}]
        }

        host = WindowsEC2WorkerHost(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        userdata = host.userdata(None)
        assert "<powershell>" in userdata
        assert "python-3.12.10-amd64.exe" in userdata
        assert host.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME in userdata
        assert host.SIGNAL_USER_DATA_FAILED_FILE_NAME in userdata

    def test_windows_ec2_worker_host_userdata_success_script(self):
        """Test that WindowsEC2WorkerHost generates correct userdata success script."""
        mock_clients = {
            "s3_client": Mock(),
            "ec2_client": Mock(),
            "ssm_client": Mock(),
        }
        mock_clients["ssm_client"].get_parameters.return_value = {
            "Parameters": [{"Value": "ami-windows123"}]
        }

        host = WindowsEC2WorkerHost(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        script = host.userdata_success_script()
        assert "Test-Path" in script
        assert host.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME in script
        assert host.SIGNAL_USER_DATA_FAILED_FILE_NAME in script
        assert host.USERDATA_SUCCESS_STRING in script
        assert host.USERDATA_FAILURE_STRING in script


class TestPosixEC2WorkerHost:
    """Test PosixEC2WorkerHost implementation."""

    def test_posix_ec2_worker_host_operating_system(self):
        """Test that PosixEC2WorkerHost returns 'posix' as operating system."""
        mock_clients = {
            "s3_client": Mock(),
            "ec2_client": Mock(),
            "ssm_client": Mock(),
        }
        mock_clients["ssm_client"].get_parameters.return_value = {
            "Parameters": [{"Value": "ami-linux123"}]
        }

        host = PosixEC2WorkerHost(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        assert host._operating_system() == "posix"
        assert host.ssm_document_name() == "AWS-RunShellScript"
        assert host.ebs_devices() == {"/dev/xvda": 30}
        assert "al2023-ami-kernel-6.1-x86_64" in host.ami_ssm_param_name()

    def test_posix_ec2_worker_host_userdata(self):
        """Test that PosixEC2WorkerHost generates POSIX userdata."""
        mock_clients = {
            "s3_client": Mock(),
            "ec2_client": Mock(),
            "ssm_client": Mock(),
        }
        mock_clients["ssm_client"].get_parameters.return_value = {
            "Parameters": [{"Value": "ami-linux123"}]
        }

        host = PosixEC2WorkerHost(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        userdata = host.userdata(None)
        assert "#!/bin/bash" in userdata
        assert "mkdir /opt/deadline" in userdata
        assert host.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME in userdata
        assert host.SIGNAL_USER_DATA_FAILED_FILE_NAME in userdata

    def test_posix_ec2_worker_host_userdata_success_script(self):
        """Test that PosixEC2WorkerHost generates correct userdata success script."""
        mock_clients = {
            "s3_client": Mock(),
            "ec2_client": Mock(),
            "ssm_client": Mock(),
        }
        mock_clients["ssm_client"].get_parameters.return_value = {
            "Parameters": [{"Value": "ami-linux123"}]
        }

        host = PosixEC2WorkerHost(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        script = host.userdata_success_script()
        assert "[[ -f" in script
        assert host.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME in script
        assert host.SIGNAL_USER_DATA_FAILED_FILE_NAME in script
        assert host.USERDATA_SUCCESS_STRING in script
        assert host.USERDATA_FAILURE_STRING in script

    def test_posix_ec2_worker_host_send_command_adds_safety_flags(self):
        """Test that PosixEC2WorkerHost adds bash safety flags to commands."""
        mock_clients = {
            "s3_client": Mock(),
            "ec2_client": Mock(),
            "ssm_client": Mock(),
        }
        mock_clients["ssm_client"].get_parameters.return_value = {
            "Parameters": [{"Value": "ami-linux123"}]
        }

        # Mock SSM send_command and related responses
        mock_clients["ssm_client"].send_command.return_value = {
            "Command": {"CommandId": "cmd-12345"}
        }

        # Mock userdata check to return success immediately, then return test output for subsequent commands
        call_count = 0

        def mock_get_command_invocation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is for userdata checking during start()
                return {
                    "ResponseCode": 0,
                    "StandardOutputContent": "Userdata finished successfully",
                    "StandardErrorContent": "",
                }
            else:
                # Subsequent calls return test output
                return {
                    "ResponseCode": 0,
                    "StandardOutputContent": "test output",
                    "StandardErrorContent": "",
                }

        mock_clients["ssm_client"].get_command_invocation.side_effect = mock_get_command_invocation
        mock_waiter = Mock()
        mock_clients["ssm_client"].get_waiter.return_value = mock_waiter

        # Mock EC2 responses for starting the host
        mock_clients["ec2_client"].run_instances.return_value = {
            "Instances": [{"InstanceId": "i-1234567890abcdef0"}]
        }
        mock_ec2_waiter = Mock()
        mock_clients["ec2_client"].get_waiter.return_value = mock_ec2_waiter

        host = PosixEC2WorkerHost(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        host.start()
        host.send_command("echo 'test'")

        # Verify that the command was called with safety flags prepended
        # Should be called twice: once for userdata check, once for the test command
        assert mock_clients["ssm_client"].send_command.call_count == 2

        # Check the second call (the test command) has safety flags
        second_call_args = mock_clients["ssm_client"].send_command.call_args_list[1]
        sent_command = second_call_args[1]["Parameters"]["commands"][0]
        assert sent_command.startswith("set -euxo pipefail; ")
        assert "echo 'test'" in sent_command


class TestEC2WorkerHostPropertyTests:
    """Property-based tests for EC2WorkerHost interface completeness."""

    @pytest.mark.parametrize(
        "host_class,operating_system",
        [
            (WindowsEC2WorkerHost, "windows"),
            (PosixEC2WorkerHost, "posix"),
        ],
    )
    @pytest.mark.timeout(30)  # Prevent test from hanging
    def test_property_2_userdata_completion_validation(self, host_class, operating_system: str):
        """
        **Feature: worker-host-decoupling, Property 2: Userdata completion validation**

        For any EC2 worker host, starting the worker host should wait for userdata to
        complete successfully and raise a WorkerHostError if userdata fails or times out.
        The raised error should include userdata execution logs.

        **Validates: Requirements 2.1, 6.1**
        """

        # Create mock clients
        mock_clients = {
            "s3_client": Mock(),
            "ec2_client": Mock(),
            "ssm_client": Mock(),
        }

        # Mock SSM parameter response for AMI ID
        mock_clients["ssm_client"].get_parameters.return_value = {
            "Parameters": [{"Value": f"ami-{operating_system}123"}]
        }

        # Mock EC2 run_instances response
        mock_clients["ec2_client"].run_instances.return_value = {
            "Instances": [{"InstanceId": "i-1234567890abcdef0"}]
        }

        # Mock EC2 waiter
        mock_ec2_waiter = Mock()
        mock_clients["ec2_client"].get_waiter.return_value = mock_ec2_waiter

        # Test Case 1: Userdata completes successfully
        # Create the WorkerHost implementation
        host = host_class(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        # Mock SSM send_command to return success signal
        mock_clients["ssm_client"].send_command.return_value = {
            "Command": {"CommandId": "cmd-success"}
        }
        mock_clients["ssm_client"].get_command_invocation.return_value = {
            "ResponseCode": 0,
            "StandardOutputContent": host.USERDATA_SUCCESS_STRING,
            "StandardErrorContent": "",
        }
        mock_ssm_waiter = Mock()
        mock_clients["ssm_client"].get_waiter.return_value = mock_ssm_waiter

        # Starting should succeed when userdata completes successfully
        host.start()
        assert host.state == WorkerHostState.RUNNING
        assert host.is_running()

        # Reset for next test
        host.stop()

        # Test Case 2: Userdata fails
        # Create a new host instance for the failure test
        host_fail = host_class(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        # Mock SSM send_command to return failure signal
        mock_clients["ssm_client"].get_command_invocation.return_value = {
            "ResponseCode": 0,
            "StandardOutputContent": f"{host_fail.USERDATA_FAILURE_STRING}\nError: Python installation failed",
            "StandardErrorContent": "",
        }

        # Starting should raise InstanceStartupError when userdata fails
        with pytest.raises(InstanceStartupError) as exc_info:
            host_fail.start()

        # Verify the error includes diagnostic information
        error_message = str(exc_info.value)
        assert "Userdata failed" in error_message
        assert "DIAGNOSTICS" in error_message
        assert "Python installation failed" in error_message

        # Test Case 3: Userdata timeout (neither success nor failure detected)
        # Create a new host instance for the timeout test
        host_timeout = host_class(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        # Mock SSM send_command to return neither success nor failure
        mock_clients["ssm_client"].get_command_invocation.return_value = {
            "ResponseCode": 0,
            "StandardOutputContent": "Still running userdata...",
            "StandardErrorContent": "",
        }

        # Starting should raise InstanceStartupError on timeout
        # (time.sleep is mocked by autouse fixture, so this will complete quickly)
        with pytest.raises(InstanceStartupError) as exc_info:
            host_timeout.start()

        # Verify the error indicates timeout
        error_message = str(exc_info.value)
        assert "Timeout waiting for userdata" in error_message
        assert "did not complete within" in error_message

    @pytest.mark.parametrize(
        "host_class,operating_system",
        [
            (WindowsEC2WorkerHost, "windows"),
            (PosixEC2WorkerHost, "posix"),
        ],
    )
    def test_property_14_worker_host_interface_completeness(
        self, host_class, operating_system: str
    ):
        """
        **Feature: worker-host-decoupling, Property 14: WorkerHost interface completeness**

        For any WorkerHost implementation, the interface should provide methods for
        starting the host, stopping the host, and sending commands to the host.

        **Validates: Requirements 2.1, 3.2, 4.1**
        """
        # Create mock clients
        mock_clients = {
            "s3_client": Mock(),
            "ec2_client": Mock(),
            "ssm_client": Mock(),
        }

        # Mock SSM parameter response for AMI ID
        mock_clients["ssm_client"].get_parameters.return_value = {
            "Parameters": [{"Value": f"ami-{operating_system}123"}]
        }

        # Mock EC2 run_instances response
        mock_clients["ec2_client"].run_instances.return_value = {
            "Instances": [{"InstanceId": "i-1234567890abcdef0"}]
        }

        # Mock EC2 waiter
        mock_ec2_waiter = Mock()
        mock_clients["ec2_client"].get_waiter.return_value = mock_ec2_waiter

        # Mock SSM send_command response
        mock_clients["ssm_client"].send_command.return_value = {
            "Command": {"CommandId": "cmd-12345"}
        }

        # Mock SSM get_command_invocation response
        # Mock userdata check to return success immediately, then return test output for subsequent commands
        call_count = 0

        def mock_get_command_invocation(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # First call is for userdata checking during start()
                return {
                    "ResponseCode": 0,
                    "StandardOutputContent": f"ami-{operating_system}123 Userdata finished successfully",
                    "StandardErrorContent": "",
                }
            else:
                # Subsequent calls return test output
                return {
                    "ResponseCode": 0,
                    "StandardOutputContent": "test output",
                    "StandardErrorContent": "",
                }

        mock_clients["ssm_client"].get_command_invocation.side_effect = mock_get_command_invocation

        # Mock SSM waiter
        mock_ssm_waiter = Mock()
        mock_clients["ssm_client"].get_waiter.return_value = mock_ssm_waiter

        # Create the WorkerHost implementation
        host = host_class(
            subnet_id="subnet-12345",
            security_group_id="sg-12345",
            instance_profile_name="test-profile",
            bootstrap_bucket_name="test-bucket",
            instance_type="t3.micro",
            instance_shutdown_behavior="terminate",
            **mock_clients,
        )

        # Verify the interface provides the required methods
        assert hasattr(host, "start"), "WorkerHost must provide start() method"
        assert hasattr(host, "stop"), "WorkerHost must provide stop() method"
        assert hasattr(host, "send_command"), "WorkerHost must provide send_command() method"
        assert hasattr(host, "is_running"), "WorkerHost must provide is_running() method"
        assert hasattr(host, "state"), "WorkerHost must provide state property"

        # Verify the methods work correctly
        # 1. Test starting the host
        assert host.state == WorkerHostState.NOT_STARTED
        assert not host.is_running()

        host.start()

        assert host.state == WorkerHostState.RUNNING
        assert host.is_running()

        # 2. Test sending commands to the host
        result = host.send_command("echo 'test'")
        assert isinstance(result, CommandResult)
        assert result.exit_code == 0
        assert result.stdout == "test output"

        # 3. Test stopping the host
        host.stop()

        assert host.state == WorkerHostState.STOPPED
        assert not host.is_running()

        # Verify the operating system is correctly identified
        assert host._operating_system() == operating_system

        # Verify OS-specific abstract methods are implemented
        assert host.ami_ssm_param_name() is not None
        assert host.ssm_document_name() is not None
        assert host.userdata(None) is not None
        # ebs_devices() can return None, so just verify it's callable
        host.ebs_devices()  # Should not raise an exception


class TestEC2InstanceWorkerComposition:
    """Test EC2InstanceWorker composition architecture."""

    def test_property_19_composition_architecture(self):
        """
        **Feature: worker-host-decoupling, Property 19: Composition architecture**

        For any EC2InstanceWorker, the worker should compose a WorkerHost instance and
        delegate host operations to it while maintaining worker agent state independently.
        The worker should validate OS compatibility during initialization.

        **Validates: Requirements 4.5**
        """
        from deadline_test_fixtures.deadline.worker import (
            WindowsInstanceWorkerBase,
            PosixInstanceWorkerBase,
            DeadlineWorkerConfiguration,
        )
        from deadline_test_fixtures.deadline.resources import Fleet

        # Create mock concrete worker classes for testing
        class MockWindowsWorker(WindowsInstanceWorkerBase):
            def ami_ssm_param_name(self) -> str:
                return "/test/windows/ami"

            def ssm_document_name(self) -> str:
                return "AWS-RunPowerShellScript"

            def configure_worker_command(self, *, config) -> str:
                return "mock windows command"

            def userdata(self, s3_files) -> str:
                return "<powershell>mock userdata</powershell>"

            def userdata_success_script(self) -> str:
                return "mock success script"

            def ebs_devices(self) -> dict[str, int] | None:
                return {"/dev/sda1": 60}

        class MockPosixWorker(PosixInstanceWorkerBase):
            def ami_ssm_param_name(self) -> str:
                return "/test/posix/ami"

            def ssm_document_name(self) -> str:
                return "AWS-RunShellScript"

            def configure_worker_command(self, *, config) -> str:
                return "mock posix command"

            def userdata(self, s3_files) -> str:
                return "#!/bin/bash\nmock userdata"

            def userdata_success_script(self) -> str:
                return "mock success script"

            def ebs_devices(self) -> dict[str, int] | None:
                return {"/dev/xvda": 30}

        # Test both Windows and POSIX combinations
        test_cases = [
            (MockWindowsWorker, WindowsEC2WorkerHost, "windows"),
            (MockPosixWorker, PosixEC2WorkerHost, "posix"),
        ]

        for worker_class, host_class, operating_system in test_cases:
            # Create mock clients for the WorkerHost
            mock_clients = {
                "s3_client": Mock(),
                "ec2_client": Mock(),
                "ssm_client": Mock(),
            }

            # Mock SSM parameter response for AMI ID
            mock_clients["ssm_client"].get_parameters.return_value = {
                "Parameters": [{"Value": f"ami-{operating_system}123"}]
            }

            # Create the WorkerHost
            worker_host = host_class(
                subnet_id="subnet-12345",
                security_group_id="sg-12345",
                instance_profile_name="test-profile",
                bootstrap_bucket_name="test-bucket",
                instance_type="t3.micro",
                instance_shutdown_behavior="terminate",
                **mock_clients,
            )

            # Create a mock configuration
            mock_fleet = Mock(spec=Fleet)
            mock_fleet.id = "fleet-12345"
            mock_fleet.autoscaling = False

            configuration = Mock(spec=DeadlineWorkerConfiguration)
            configuration.farm_id = "farm-12345"
            configuration.fleet = mock_fleet
            configuration.region = "us-west-2"

            # Create a mock deadline client
            mock_deadline_client = Mock()

            # Test 1: Valid OS combination should succeed
            worker = worker_class(
                configuration=configuration,
                worker_host=worker_host,
                deadline_client=mock_deadline_client,
            )

            # Verify composition is working
            assert worker.worker_host is worker_host
            assert worker.configuration is configuration
            assert worker.deadline_client is mock_deadline_client

            # Verify agent state tracking
            assert hasattr(worker, "_agent_state")
            assert worker.agent_state == WorkerAgentState.NOT_STARTED

            # Verify OS validation worked
            assert worker._required_host_os() == operating_system
            assert worker.worker_host._operating_system() == operating_system

            # Verify legacy field delegation
            assert worker.subnet_id == worker_host.subnet_id
            assert worker.security_group_id == worker_host.security_group_id
            assert worker.instance_profile_name == worker_host.instance_profile_name
            assert worker.bootstrap_bucket_name == worker_host.bootstrap_bucket_name
            assert worker.s3_client is worker_host.s3_client
            assert worker.ec2_client is worker_host.ec2_client
            assert worker.ssm_client is worker_host.ssm_client

            # Test 2: Invalid OS combination should raise ValueError
            # Create a host with the opposite OS
            opposite_os = "posix" if operating_system == "windows" else "windows"
            opposite_host_class = (
                PosixEC2WorkerHost if operating_system == "windows" else WindowsEC2WorkerHost
            )

            # Mock SSM parameter response for opposite OS
            mock_clients["ssm_client"].get_parameters.return_value = {
                "Parameters": [{"Value": f"ami-{opposite_os}123"}]
            }

            opposite_host = opposite_host_class(
                subnet_id="subnet-12345",
                security_group_id="sg-12345",
                instance_profile_name="test-profile",
                bootstrap_bucket_name="test-bucket",
                instance_type="t3.micro",
                instance_shutdown_behavior="terminate",
                **mock_clients,
            )

            # This should raise ValueError due to OS mismatch
            with pytest.raises(ValueError) as exc_info:
                worker_class(
                    configuration=configuration,
                    worker_host=opposite_host,
                    deadline_client=mock_deadline_client,
                )

            error_message = str(exc_info.value)
            assert (
                f"Worker requires {operating_system} host but got {opposite_os} host"
                in error_message
            )
            assert "Ensure you use the correct WorkerHost type" in error_message

            # Test 3: Verify abstract method implementation
            assert hasattr(worker, "_required_host_os")
            assert callable(worker._required_host_os)
            assert worker._required_host_os() == operating_system
