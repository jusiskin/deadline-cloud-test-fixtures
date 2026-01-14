# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
from __future__ import annotations

import abc
import json
import logging
import os
import pathlib
import posixpath
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import InitVar, dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar, Dict, Optional, cast

import botocore.client
import botocore.exceptions

from ..models import (
    PipInstall,
    PosixSessionUser,
)
from .resources import CloudWatchLogEvent, Fleet, WorkerLog
from .worker_host import Ec2Tag, CommandResult, EC2WorkerHost, WorkerAgentState
from ..util import call_api, wait_for

if TYPE_CHECKING:
    from botocore.paginate import PageIterator, Paginator

LOG = logging.getLogger(__name__)

DOCKER_CONTEXT_DIR = os.path.join(os.path.dirname(__file__), "..", "containers", "worker")

DEFAULT_WAITER_CONFIG = {
    "Delay": 5,
    "MaxAttempts": 30,
}


class DeadlineWorker(abc.ABC):
    @abc.abstractmethod
    def start(self) -> None:
        pass

    @abc.abstractmethod
    def stop(self) -> None:
        pass

    @abc.abstractmethod
    def send_command(self, command: str) -> CommandResult:
        pass

    @abc.abstractmethod
    def get_worker_id(self) -> str:
        pass


@dataclass(frozen=True)
class WorkerLogConfig:
    cloudwatch_log_group: str
    """The name of the CloudWatch Log Group that the Agent log should be streamed to"""

    cloudwatch_log_stream: str
    """The name of the CloudWatch Log Stream that the Agent log should be streamed to"""


class InstanceStartupError(Exception):
    """Custom exception for instance startup failures with diagnostics."""

    def __init__(self, message, diagnostics=None):
        self.message = message
        self.diagnostics = diagnostics

        # Format a more visually distinct error message with the diagnostics
        error_msg = [
            f"{message}",
            "=" * 80,  # Separator line
            "DIAGNOSTICS",
            "=" * 80,  # Separator line
        ]

        if diagnostics:
            error_msg.append(str(diagnostics))
        else:
            error_msg.append("No diagnostics available")

        error_msg.append("=" * 80)  # Separator line

        super().__init__("\n".join(error_msg))


class WorkerHostError(Exception):
    """
    Exception raised when worker host operations fail.

    This exception includes diagnostic information about the worker host state
    to help identify the source of the failure.

    Attributes:
        message: Human-readable error message
        host_os: Operating system of the worker host (e.g., 'windows', 'posix')
        instance_id: EC2 instance ID (if available)
        diagnostics: Additional diagnostic information about the host state
    """

    def __init__(
        self,
        message: str,
        host_os: Optional[str] = None,
        instance_id: Optional[str] = None,
        diagnostics: Optional[str] = None,
    ):
        self.message = message
        self.host_os = host_os
        self.instance_id = instance_id
        self.diagnostics = diagnostics

        # Format error message with clear indication this is a host-level error
        error_msg = [
            "WORKER HOST ERROR",
            "=" * 80,
            f"{message}",
            "=" * 80,
            "HOST DIAGNOSTICS",
            "=" * 80,
        ]

        if host_os:
            error_msg.append(f"Operating System: {host_os}")
        if instance_id:
            error_msg.append(f"Instance ID: {instance_id}")
        if diagnostics:
            error_msg.append(f"\n{diagnostics}")
        else:
            error_msg.append("No additional diagnostics available")

        error_msg.append("=" * 80)

        super().__init__("\n".join(error_msg))


class WorkerAgentError(Exception):
    """
    Exception raised when worker agent operations fail.

    This exception includes diagnostic information about the worker agent state
    and relevant log output to help identify the source of the failure.

    Attributes:
        message: Human-readable error message
        configuration: Worker agent configuration (if available)
        command_result: Result of the failed command (if applicable)
        logs: Relevant log output from the worker agent
        worker_id: Worker ID (if available)
    """

    def __init__(
        self,
        message: str,
        configuration: Optional[DeadlineWorkerConfiguration] = None,
        command_result: Optional[CommandResult] = None,
        logs: Optional[str] = None,
        worker_id: Optional[str] = None,
    ):
        self.message = message
        self.configuration = configuration
        self.command_result = command_result
        self.logs = logs
        self.worker_id = worker_id

        # Format error message with clear indication this is an agent-level error
        error_msg = [
            "WORKER AGENT ERROR",
            "=" * 80,
            f"{message}",
            "=" * 80,
            "AGENT DIAGNOSTICS",
            "=" * 80,
        ]

        if worker_id:
            error_msg.append(f"Worker ID: {worker_id}")
        if configuration:
            error_msg.append(f"Farm ID: {configuration.farm_id}")
            error_msg.append(f"Fleet ID: {configuration.fleet.id}")
            error_msg.append(f"Region: {configuration.region}")
        if command_result:
            error_msg.append(f"\nCommand Exit Code: {command_result.exit_code}")
            if command_result.stdout:
                error_msg.append(f"Command Output:\n{command_result.stdout}")
            if command_result.stderr:
                error_msg.append(f"Command Error:\n{command_result.stderr}")
        if logs:
            error_msg.append(f"\nAgent Logs:\n{logs}")
        else:
            error_msg.append("No additional diagnostics available")

        error_msg.append("=" * 80)

        super().__init__("\n".join(error_msg))


@dataclass(frozen=True)
class DeadlineWorkerConfiguration:
    farm_id: str
    fleet: Fleet
    region: str
    allow_shutdown: bool
    worker_agent_install: PipInstall
    start_service: bool = True
    no_install_service: bool = False
    service_model_path: str | None = None
    no_local_session_logs: str | None = None
    disallow_instance_profile: str | None = None

    file_mappings: list[tuple[str, str]] | None = None
    """Mapping of files to copy from host environment to worker environment"""

    pre_install_commands: list[str] | None = None
    """Commands to run before installing the Worker agent"""

    job_user: str = field(default="job-user")
    agent_user: str = field(default="deadline-worker")
    windows_user_secret: str | None = None
    job_user_group: str = field(default="deadline-job-users")

    job_users: list[PosixSessionUser] = field(
        default_factory=lambda: [PosixSessionUser("job-user", "job-user")]
    )
    """Additional job users to configure for Posix workers"""

    windows_job_users: list = field(default_factory=lambda: ["job-user"])
    """Additional job users to configure for Windows workers"""

    session_root_dir: str | None = None
    """Path to parent directory of worker session directories"""

    worker_env_var: Dict[str, str] | None = None
    """Additional feature flag to configure for workers"""


@dataclass
class EC2InstanceWorker(DeadlineWorker):
    """
    EC2-based worker with composed WorkerHost.

    Args:
        configuration: Worker agent configuration (farm, fleet, region, etc.)
        worker_host: The EC2 worker host to use (required)

    Example:
        >>> host = WindowsEC2WorkerHost(subnet_id="...", ...)
        >>> host.start()
        >>> worker = WindowsInstanceBuildWorker(configuration=config, worker_host=host)
        >>> worker.start()
        >>> # Use worker...
        >>> worker.stop()
        >>> # Host is still running and can be reused
    """

    configuration: DeadlineWorkerConfiguration
    worker_host: EC2WorkerHost
    deadline_client: botocore.client.BaseClient
    worker_id: Optional[str] = field(init=False, default=None)
    _agent_state: WorkerAgentState = field(init=False, default=WorkerAgentState.NOT_STARTED)

    # Legacy fields for backward compatibility - will be removed in future tasks
    subnet_id: str = field(init=False)
    security_group_id: str = field(init=False)
    instance_profile_name: str = field(init=False)
    bootstrap_bucket_name: str = field(init=False)
    s3_client: botocore.client.BaseClient = field(init=False)
    ec2_client: botocore.client.BaseClient = field(init=False)
    ssm_client: botocore.client.BaseClient = field(init=False)
    instance_type: str = field(init=False)
    instance_shutdown_behavior: str = field(init=False)
    additional_tags: list[Ec2Tag] = field(init=False, default_factory=list)

    USERDATA_SUCCESS_STRING: ClassVar[str] = "Userdata finished successfully"
    USERDATA_FAILURE_STRING: ClassVar[str] = "Userdata failed to finish"

    """
    Option to override the AMI ID for the EC2 instance. If no override is provided, the default will depend on the subclass being instansiated.
    """
    override_ami_id: InitVar[Optional[str]] = None

    def __post_init__(self, override_ami_id: Optional[str] = None):
        """Initialize the worker and validate operating system compatibility."""
        # Validate that the worker host OS matches the worker requirements
        required_os = self._required_host_os()
        host_os = self.worker_host._operating_system()
        if required_os != host_os:
            raise ValueError(
                f"Worker requires {required_os} host but got {host_os} host. "
                f"Ensure you use the correct WorkerHost type for this worker."
            )

        # Set legacy fields from worker_host for backward compatibility
        self.subnet_id = self.worker_host.subnet_id
        self.security_group_id = self.worker_host.security_group_id
        self.instance_profile_name = self.worker_host.instance_profile_name
        self.bootstrap_bucket_name = self.worker_host.bootstrap_bucket_name
        self.s3_client = self.worker_host.s3_client
        self.ec2_client = self.worker_host.ec2_client
        self.ssm_client = self.worker_host.ssm_client
        self.instance_type = self.worker_host.instance_type
        self.instance_shutdown_behavior = self.worker_host.instance_shutdown_behavior
        self.additional_tags = self.worker_host.additional_tags
        self.additional_tags = self.worker_host.additional_tags

        if override_ami_id:
            self._ami_id = override_ami_id

    @property
    def agent_state(self) -> WorkerAgentState:
        """Get the current state of the worker agent."""
        return self._agent_state

    @property
    def instance_id(self) -> Optional[str]:
        """Get the instance ID from the worker host."""
        return self.worker_host.instance_id

    @abc.abstractmethod
    def _required_host_os(self) -> str:
        """Return the required host operating system (e.g., 'windows', 'posix')."""
        pass

    @abc.abstractmethod
    def _install_agent(self) -> None:
        """Install worker agent software (OS-specific)."""
        pass

    @abc.abstractmethod
    def _configure_agent(self) -> None:
        """Configure worker agent (OS-specific)."""
        pass

    @abc.abstractmethod
    def _start_agent_service(self) -> None:
        """Start the worker agent service (OS-specific)."""
        pass

    @abc.abstractmethod
    def _stop_agent_service(self) -> None:
        """Stop the worker agent service (OS-specific)."""
        pass

    @abc.abstractmethod
    def _cleanup_agent_state(self) -> None:
        """Clean up worker agent state files (OS-specific)."""
        pass

    def _transfer_files(self) -> None:
        """Transfer file_mappings from local machine to the EC2 instance."""
        if not self.configuration.file_mappings:
            return

        # Delegate to worker_host for complete file transfer (local -> S3 -> EC2)
        self.worker_host.transfer_files(self.configuration.file_mappings)

    def _cleanup_files(self) -> None:
        """Clean up files that were staged for this worker agent."""
        if not self.configuration.file_mappings:
            return

        # Collect all destination file paths
        file_paths = [dst_path for _, dst_path in self.configuration.file_mappings]

        # Delegate to worker_host for file cleanup
        self.worker_host.cleanup_files(file_paths)

    def _delete_worker(self) -> None:
        """Delete the worker from Deadline Cloud service."""
        if not self.worker_id:
            LOG.info("No worker_id available, skipping worker deletion")
            return

        if not self.configuration.fleet.autoscaling:
            try:
                self.wait_until_stopped()
            except TimeoutError:
                LOG.warning(
                    f"{self.worker_id} did not transition to a STOPPED status, forcibly stopping..."
                )
                self.set_stopped_status()

            try:
                self.delete()
            except botocore.exceptions.ClientError as error:
                LOG.exception(f"Failed to delete worker: {error}")
                raise

    @abc.abstractmethod
    def configure_worker_command(
        self, *, config: DeadlineWorkerConfiguration
    ) -> str:  # pragma: no cover
        raise NotImplementedError("'configure_worker_command' was not implemented.")

    @abc.abstractmethod
    def start_worker_service(self) -> None:  # pragma: no cover
        raise NotImplementedError("'_start_worker_service' was not implemented.")

    @abc.abstractmethod
    def stop_worker_service(self) -> None:  # pragma: no cover
        raise NotImplementedError("'_stop_worker_service' was not implemented.")

    @abc.abstractmethod
    def get_worker_id(self) -> str:
        raise NotImplementedError("'get_worker_id' was not implemented.")

    def start(self) -> None:
        """
        Install, configure, and start the worker agent (assumes host is already running).

        This method performs the complete worker agent startup:
        1. Validate the worker host is running
        2. Claim the worker host for this worker (prevents other workers from using it)
        3. Stage files to S3 if needed
        4. Install worker agent software on the host
        5. Configure the worker agent with the provided configuration (farm, fleet, region, etc.)
        6. Start the worker agent service
        7. Retrieve and store the worker ID

        Raises:
            RuntimeError: If the worker host is not running
            RuntimeError: If this worker already has an agent running
            RuntimeError: If another worker already has an agent on this host
        """
        if not self.worker_host.is_running():
            raise WorkerAgentError(
                message="Cannot start worker agent: worker host is not running",
                configuration=self.configuration,
                logs="Call worker_host.start() first to start the host before starting the agent.",
            )

        if self._agent_state == WorkerAgentState.RUNNING:
            raise WorkerAgentError(
                message="Cannot start worker agent: this worker already has an agent running",
                worker_id=self.worker_id,
                configuration=self.configuration,
                logs="Call stop() first to remove the existing agent before starting a new one.",
            )

        # Claim the host for this worker (raises if another worker is using it)
        self.worker_host._claim_for_worker(id(self))

        # Transfer files from local machine to EC2 instance
        self._transfer_files()

        # Install, configure, and start the worker agent
        self._install_agent()
        self._configure_agent()
        self._start_agent_service()
        self.worker_id = self.get_worker_id()
        self._agent_state = WorkerAgentState.RUNNING

    def stop(self) -> None:
        """
        Stop the worker agent and remove all agent resources (leaves host running).

        This method performs complete worker agent teardown:
        1. Stop the worker agent service
        2. Remove worker agent state files (worker.json, configuration files, etc.)
        3. Clean up staged files from file_mappings
        4. Delete the worker from Deadline Cloud service
        5. Clear the worker ID
        6. Release the worker host so other workers can use it

        After this method completes, the host is ready for a new worker agent configuration.
        """
        if self._agent_state == WorkerAgentState.NOT_STARTED:
            # Nothing to clean up
            return

        if self.worker_id:
            self._stop_agent_service()
            self._cleanup_agent_state()
            self._cleanup_files()
            self._delete_worker()
            self.worker_id = None

        # Release the host so other workers can use it
        self.worker_host._release_from_worker(id(self))
        self._agent_state = WorkerAgentState.NOT_STARTED

    def delete(self):
        try:
            self.deadline_client.delete_worker(
                farmId=self.configuration.farm_id,
                fleetId=self.configuration.fleet.id,
                workerId=self.worker_id,
            )
            LOG.info(f"{self.worker_id} has been deleted from {self.configuration.fleet.id}")
        except botocore.exceptions.ClientError as error:
            LOG.exception(f"Failed to delete worker: {error}")
            raise

    def wait_until_stopped(
        self, *, max_checks: int = 25, seconds_between_checks: float = 5
    ) -> None:
        self.wait_until_desired_worker_status(
            max_checks=max_checks,
            seconds_between_checks=seconds_between_checks,
            desired_status="STOPPED",
        )

    def wait_until_desired_worker_status(
        self,
        *,
        max_checks: int = 25,
        seconds_between_checks: float = 5,
        desired_status: str = "STOPPED",
    ) -> None:
        for _ in range(max_checks):
            response = self.deadline_client.get_worker(
                farmId=self.configuration.farm_id,
                fleetId=self.configuration.fleet.id,
                workerId=self.worker_id,
            )
            if response["status"] == desired_status:
                LOG.info(f"{self.worker_id} is {desired_status}")
                break
            time.sleep(seconds_between_checks)
            LOG.info(f"Waiting for {self.worker_id} to transition to {desired_status} status")
        else:
            raise TimeoutError

    def set_stopped_status(self):
        LOG.info(f"Setting {self.worker_id} to STOPPED status")
        try:
            self.deadline_client.update_worker(
                farmId=self.configuration.farm_id,
                fleetId=self.configuration.fleet.id,
                workerId=self.worker_id,
                status="STOPPED",
            )
        except botocore.exceptions.ClientError as error:
            LOG.exception(f"Failed to update worker status: {error}")
            raise

    def _get_worker_logs(self) -> Optional[WorkerLogConfig]:
        """Get the log group and log stream for the worker. Retain the API structure"""
        response = self.deadline_client.get_worker(
            farmId=self.configuration.farm_id,
            fleetId=self.configuration.fleet.id,
            workerId=self.worker_id,
        )
        if log_config := response.get("log"):
            LOG.info(f"Log Config structure {log_config}")
            if log_config_options := log_config.get("options"):
                log_group_name = log_config_options.get("logGroupName")
                log_stream_name = log_config_options.get("logStreamName")
                if log_group_name and log_stream_name:
                    return WorkerLogConfig(
                        cloudwatch_log_group=log_group_name, cloudwatch_log_stream=log_stream_name
                    )
        # Default, no log config yet.
        return None

    def get_logs(self, *, logs_client: botocore.client.BaseClient) -> WorkerLog:
        # Get the worker log group and stream from the service.
        log_config: Optional[WorkerLogConfig] = self._get_worker_logs()
        if not log_config:
            return WorkerLog(worker_id=self.worker_id, logs=[])  # type: ignore[arg-type]

        filter_log_events_paginator: Paginator = logs_client.get_paginator("filter_log_events")
        filter_log_events_pages: PageIterator = call_api(
            description=f"Fetching log events for worker {self.worker_id} in log group {log_config.cloudwatch_log_group}",
            fn=lambda: filter_log_events_paginator.paginate(
                logGroupName=log_config.cloudwatch_log_group,
                logStreamNames=[log_config.cloudwatch_log_stream],
            ),
        )
        log_events = filter_log_events_pages.build_full_result()
        log_events = [CloudWatchLogEvent.from_api_response(e) for e in log_events["events"]]
        # For debugging test cases.
        # LOG.info(log_events)

        return WorkerLog(worker_id=self.worker_id, logs=log_events)  # type: ignore[arg-type]

    def send_command(self, command: str) -> CommandResult:
        """Delegate to worker host."""
        return self.worker_host.send_command(command)


@dataclass
class WindowsInstanceWorkerBase(EC2InstanceWorker):
    """
    Base class for Windows EC2 workers.

    Args:
        configuration: Worker agent configuration
        worker_host: WindowsEC2WorkerHost to use (required)

    Example:
        >>> host = WindowsEC2WorkerHost(subnet_id="...", security_group_id="...", ...)
        >>> host.start()
        >>> worker = WindowsInstanceBuildWorker(configuration=config, worker_host=host)
        >>> worker.start()
    """

    def _required_host_os(self) -> str:
        """Windows workers require Windows hosts."""
        return "windows"

    def send_command(
        self, command: str, ssm_waiter_config: dict[str, int] = DEFAULT_WAITER_CONFIG
    ) -> CommandResult:
        """Delegate to worker host with Windows-specific command handling."""
        return self.worker_host.send_command(command, ssm_waiter_config)

    def _install_agent(self) -> None:
        """Install worker agent software on Windows."""
        # Installation is handled in _configure_agent for Windows
        pass

    def _configure_agent(self) -> None:
        """Configure worker agent on Windows."""
        assert self.instance_id
        LOG.info(f"Sending SSM command to configure Worker agent on instance {self.instance_id}")

        cmd_result = self.send_command(
            f"{self.configure_worker_command(config=self.configuration)}",
            {"Delay": 5, "MaxAttempts": 48},
        )
        if cmd_result.exit_code != 0:
            raise WorkerAgentError(
                message="Failed to configure Worker agent. Worker agent configuration command failed. Check command output for details.",
                configuration=self.configuration,
                command_result=cmd_result,
            )
        LOG.info("Successfully configured Worker agent")

    def _start_agent_service(self) -> None:
        """Start the worker agent service."""
        if self.configuration.start_service:
            LOG.info(
                f"Sending SSM command to start Windows Worker agent on instance {self.instance_id}"
            )
            self.start_worker_service()
            LOG.info("Successfully started Worker agent")

    def _stop_agent_service(self) -> None:
        """Stop the worker agent service."""
        self.stop_worker_service()

    def _cleanup_agent_state(self) -> None:
        """Clean up worker agent state files on Windows."""
        # Remove worker state files
        cleanup_commands = [
            'Remove-Item -Path "C:\\ProgramData\\Amazon\\Deadline\\Cache\\worker.json" -Force -ErrorAction SilentlyContinue',
            'Remove-Item -Path "C:\\ProgramData\\Amazon\\Deadline\\Logs\\*" -Force -Recurse -ErrorAction SilentlyContinue',
        ]

        for cmd in cleanup_commands:
            try:
                self.send_command(cmd)
            except Exception as e:
                LOG.warning(f"Failed to clean up agent state: {e}")

    # Abstract methods that subclasses must implement
    @abc.abstractmethod
    def configure_worker_command(self, *, config: DeadlineWorkerConfiguration) -> str:
        """Generate the command to configure the worker agent."""
        pass

    # Public methods for worker agent management
    def start_worker_service(self) -> None:
        """Start the worker agent Windows service."""
        LOG.info("Sending command to start the Worker Agent service")

        cmd_result = self.send_command(
            " ; ".join(
                [
                    'Start-Service -Name "DeadlineWorker"',
                    "echo 'Running Get-Process to check if the agent is running'",
                    'for($i=1; $i -le 30 -and "" -ne $err ; $i++){sleep $i; Get-Process pythonservice -ErrorVariable err}',
                    "IF(Get-Process pythonservice){echo '+++SERVICE IS RUNNING+++'}ELSE{echo '+++SERVICE NOT RUNNING+++'; Get-Content -Encoding utf8 C:\\ProgramData\\Amazon\\Deadline\\Logs\\worker-agent-bootstrap.log,C:\\ProgramData\\Amazon\\Deadline\\Logs\\worker-agent.log; exit 1}",
                ]
            ),
        )

        if cmd_result.exit_code != 0:
            raise WorkerAgentError(
                message="Failed to start Worker Agent service. Worker agent service failed to start. Check command output and agent logs.",
                configuration=self.configuration,
                command_result=cmd_result,
            )

        self.worker_id = self.get_worker_id()

    def stop_worker_service(self) -> None:
        """Stop the worker agent Windows service."""
        LOG.info("Sending command to stop the Worker Agent service")
        cmd_result = self.send_command('Stop-Service -Name "DeadlineWorker"')

        if cmd_result.exit_code != 0:
            raise WorkerAgentError(
                message="Failed to stop Worker Agent service. Worker agent service failed to stop. Check command output for details.",
                configuration=self.configuration,
                worker_id=self.worker_id,
                command_result=cmd_result,
            )

    def get_worker_id(self) -> str:
        """Retrieve the worker ID from the worker agent."""
        LOG.info(f"Sending SSM command to get the worker ID on instance {self.instance_id}")
        cmd_result = self.send_command(
            " ; ".join(
                [
                    'for($i=1; $i -le 20 -and "" -ne $err ; $i++){sleep $i; Get-Item C:\\ProgramData\\Amazon\\Deadline\\Cache\\worker.json -ErrorVariable err 1>$null}',
                    "$worker=Get-Content -Raw C:\\ProgramData\\Amazon\\Deadline\\Cache\\worker.json | ConvertFrom-Json",
                    "echo $worker.worker_id",
                ]
            ),
            {"Delay": 5, "MaxAttempts": 36},
        )
        if cmd_result.exit_code != 0:
            raise WorkerAgentError(
                message="Failed to get Worker ID. Could not retrieve worker ID from worker.json file. The worker agent may not have started correctly.",
                configuration=self.configuration,
                command_result=cmd_result,
            )

        worker_id = cmd_result.stdout.rstrip("\n\r")
        assert re.match(
            r"^worker-[0-9a-f]{32}$", worker_id
        ), f"Got nonvalid Worker ID from command stdout: {cmd_result}"

        LOG.info(f"Obtained Worker ID: {worker_id}")
        return worker_id

    def configure_worker_common(self, *, config: DeadlineWorkerConfiguration) -> str:
        """Get the command to configure the Worker. This must be run as Administrator.
        This cannot assume that the agent user exists.
        """

        cmds = ["$ErrorActionPreference = 'Stop'"]

        if config.service_model_path:
            cmds.append(
                f"aws configure add-model --service-model file://{config.service_model_path} --service-name deadline; "
                f"Copy-Item -Path ~\\.aws\\* -Destination C:\\Users\\Administrator\\.aws\\models -Recurse; "
                f"Copy-Item -Path ~\\.aws\\* -Destination C:\\Users\\{config.job_user}\\.aws\\models -Recurse"
            )

        if config.no_local_session_logs:
            cmds.append(
                "[System.Environment]::SetEnvironmentVariable('DEADLINE_WORKER_LOCAL_SESSION_LOGS', 'false', [System.EnvironmentVariableTarget]::Machine); "
                "$env:DEADLINE_WORKER_LOCAL_SESSION_LOGS = [System.Environment]::GetEnvironmentVariable('DEADLINE_WORKER_LOCAL_SESSION_LOGS','Machine')",
            )

        if os.environ.get("DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE"):
            LOG.info(
                f"Using DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE: {os.environ.get('DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE')}"
            )
            cmds.append(
                f"[System.Environment]::SetEnvironmentVariable('DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE', '{os.environ.get('DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE')}', [System.EnvironmentVariableTarget]::Machine); "
                "$env:DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE = [System.Environment]::GetEnvironmentVariable('DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE','Machine')",
            )

        if os.environ.get("AWS_ENDPOINT_URL_DEADLINE"):
            LOG.info(
                f"Using AWS_ENDPOINT_URL_DEADLINE: {os.environ.get('AWS_ENDPOINT_URL_DEADLINE')}"
            )
            cmds.append(
                f"[System.Environment]::SetEnvironmentVariable('AWS_ENDPOINT_URL_DEADLINE', '{os.environ.get('AWS_ENDPOINT_URL_DEADLINE')}', [System.EnvironmentVariableTarget]::Machine); "
                "$env:AWS_ENDPOINT_URL_DEADLINE = [System.Environment]::GetEnvironmentVariable('AWS_ENDPOINT_URL_DEADLINE','Machine')",
            )

        if config.worker_env_var:
            for key, value in config.worker_env_var.items():
                cmds.append(
                    f"[System.Environment]::SetEnvironmentVariable('{key}', '{value}', [System.EnvironmentVariableTarget]::Machine); "
                    f"$env:{key} = [System.Environment]::GetEnvironmentVariable('{value}','Machine')",
                )

        return "; ".join(cmds)

    def get_windows_user_secret_cmd(self, secret_id: str) -> str:
        """
        Returns a PowerShell command string that will retrieve and use the secret on the worker instance itself.

        Args:
            secret_id: The ID of the secret in Secrets Manager

        Returns:
            str: PowerShell command to fetch and extract the password from the secret
        """
        return (
            "aws secretsmanager get-secret-value "
            f"--secret-id {secret_id} "
            "--query 'SecretString' --output text | "
            "ConvertFrom-Json | "
            "Select-Object -ExpandProperty password"
        )


@dataclass
class WindowsInstanceBuildWorker(WindowsInstanceWorkerBase):
    """
    This class represents a Windows EC2 Worker Host.
    Any commands must be written in Powershell.
    """

    def configure_worker_command(self, *, config: DeadlineWorkerConfiguration) -> str:
        """Get the command to configure the Worker. This must be run as Administrator."""

        cmds = [
            "Set-PSDebug -trace 1",
            self.configure_worker_common(config=config),
            config.worker_agent_install.install_command_for_windows,
            *(config.pre_install_commands or []),
            # fmt: off
            (
                "install-deadline-worker "
                + "-y "
                + f"--farm-id {config.farm_id} "
                + f"--fleet-id {config.fleet.id} "
                + f"--region {config.region} "
                + f"--user {config.agent_user} "
                + (
                    f"--password $({self.get_windows_user_secret_cmd(secret_id=config.windows_user_secret)}) "
                    if config.windows_user_secret
                    else ""
                )
                + f"{'--allow-shutdown ' if config.allow_shutdown else ''}"
                + f"{'--disallow-instance-profile ' if config.disallow_instance_profile else ''}"
                + (
                    f"--session-root-dir {config.session_root_dir} "
                    if config.session_root_dir is not None
                    else ""
                )
            ),
            # fmt: on
        ]

        if config.service_model_path:
            cmds.append(
                f"Copy-Item -Path ~\\.aws\\* -Destination C:\\Users\\{config.agent_user}\\.aws\\models -Recurse; "
            )

        if config.start_service:
            cmds.append('Start-Service -Name "DeadlineWorker"')

        return "; ".join(cmds)


@dataclass
class PosixInstanceWorkerBase(EC2InstanceWorker):
    """
    Base class for POSIX EC2 workers.

    Args:
        configuration: Worker agent configuration
        worker_host: PosixEC2WorkerHost to use (required)

    Example:
        >>> host = PosixEC2WorkerHost(subnet_id="...", security_group_id="...", ...)
        >>> host.start()
        >>> worker = PosixInstanceBuildWorker(configuration=config, worker_host=host)
        >>> worker.start()
    """

    def _required_host_os(self) -> str:
        """POSIX workers require POSIX hosts."""
        return "posix"

    def send_command(
        self, command: str, ssm_waiter_config: dict[str, int] = DEFAULT_WAITER_CONFIG
    ) -> CommandResult:
        """Delegate to worker host with POSIX-specific command prefix."""
        return self.worker_host.send_command(command, ssm_waiter_config)

    def _install_agent(self) -> None:
        """Install worker agent software on POSIX."""
        # Installation is handled in _configure_agent for POSIX
        pass

    def _configure_agent(self) -> None:
        """Configure worker agent on POSIX."""
        assert self.instance_id
        LOG.info(
            f"Starting worker for farm: {self.configuration.farm_id} and fleet: {self.configuration.fleet.id}"
        )
        LOG.info(f"Sending SSM command to configure Worker agent on instance {self.instance_id}")

        cmd_result = self.send_command(self.configure_worker_command(config=self.configuration))
        if cmd_result.exit_code != 0:
            raise WorkerAgentError(
                message="Failed to configure Worker agent. Worker agent configuration command failed. Check command output for details.",
                configuration=self.configuration,
                command_result=cmd_result,
            )
        LOG.info("Successfully configured Worker agent")

    def _start_agent_service(self) -> None:
        """Start the worker agent service."""
        if self.configuration.start_service:
            LOG.info(
                f"Sending SSM command to configure Worker agent on instance {self.instance_id}"
            )
            self.start_worker_service()
            LOG.info("Successfully started worker agent")

    def _stop_agent_service(self) -> None:
        """Stop the worker agent service."""
        self.stop_worker_service()

    def _cleanup_agent_state(self) -> None:
        """Clean up worker agent state files on POSIX."""
        # Remove worker state files
        cleanup_commands = [
            "rm -f /var/lib/deadline/worker.json",
            "rm -rf /var/log/amazon/deadline/*",
        ]

        for cmd in cleanup_commands:
            try:
                self.send_command(cmd)
            except Exception as e:
                LOG.warning(f"Failed to clean up agent state: {e}")

    # Abstract methods that subclasses must implement
    @abc.abstractmethod
    def configure_worker_command(self, *, config: DeadlineWorkerConfiguration) -> str:
        """Generate the command to configure the worker agent."""
        pass

    # Public methods for worker agent management
    def start_worker_service(self) -> None:
        """Start the worker agent systemd service."""
        LOG.info("Sending command to start the Worker Agent service")

        cmd_result = self.send_command(
            " && ".join(
                [
                    "systemctl start deadline-worker",
                    "sleep 5",
                    "systemctl is-active deadline-worker",
                    "if test $? -ne 0; then echo '+++AGENT NOT RUNNING+++'; cat /var/log/amazon/deadline/worker-agent-bootstrap.log /var/log/amazon/deadline/worker-agent.log; exit 1; fi",
                ]
            )
        )

        if cmd_result.exit_code != 0:
            raise WorkerAgentError(
                message="Failed to start Worker Agent service. Worker agent service failed to start. Check command output and agent logs.",
                configuration=self.configuration,
                command_result=cmd_result,
            )

        self.worker_id = self.get_worker_id()

    def stop_worker_service(self) -> None:
        """Stop the worker agent systemd service."""
        LOG.info("Sending command to stop the Worker Agent service")
        cmd_result = self.send_command("systemctl stop deadline-worker")

        if cmd_result.exit_code != 0:
            raise WorkerAgentError(
                message="Failed to stop Worker Agent service. Worker agent service failed to stop. Check command output for details.",
                configuration=self.configuration,
                worker_id=self.worker_id,
                command_result=cmd_result,
            )

    def get_worker_id(self) -> str:
        """Retrieve the worker ID from the worker agent."""
        # There can be a race condition, so we may need to wait a little bit for the status file to be written.

        worker_state_filename = "/var/lib/deadline/worker.json"
        cmd_result = self.send_command(
            " && ".join(
                [
                    f"t=0 && while [ $t -le 10 ] && ! (test -f {worker_state_filename}); do sleep $t; t=$[$t+1]; done",
                    f"cat {worker_state_filename} | jq -r '.worker_id'",
                ]
            )
        )
        if cmd_result.exit_code != 0:
            raise WorkerAgentError(
                message="Failed to get Worker ID. Could not retrieve worker ID from worker.json file. The worker agent may not have started correctly.",
                configuration=self.configuration,
                command_result=cmd_result,
            )

        worker_id = cmd_result.stdout.rstrip("\n\r")
        LOG.info(f"Worker ID: {worker_id}")
        assert re.match(
            r"^worker-[0-9a-f]{32}$", worker_id
        ), f"Got nonvalid Worker ID from command stdout: {cmd_result}"
        return worker_id

    def configure_agent_user_environment(
        self, config: DeadlineWorkerConfiguration
    ) -> str:  # pragma: no cover
        """Get the command to configure the Worker. This must be run as root.
        This can assume that the agent user exists.
        """

        cmds = []

        if config.service_model_path:
            cmds.append(
                f"runuser -l {config.agent_user} -s /bin/bash -c 'aws configure add-model --service-model file://{config.service_model_path}'"
            )

        allow_instance_profile = os.environ.get("DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE", None)
        endpoint_url_deadline = os.environ.get("AWS_ENDPOINT_URL_DEADLINE", None)

        # Create a systemd drop-in config file to apply the configuration
        # See https://wiki.archlinux.org/title/Systemd#Drop-in_files
        cmds.extend(
            [
                "mkdir -p /etc/systemd/system/deadline-worker.service.d/",
                'echo "[Service]" >> /etc/systemd/system/deadline-worker.service.d/config.conf',
                # Configure the region
                f'echo "Environment=AWS_REGION={config.region}" >> /etc/systemd/system/deadline-worker.service.d/config.conf',
                f'echo "Environment=AWS_DEFAULT_REGION={config.region}" >> /etc/systemd/system/deadline-worker.service.d/config.conf',
            ]
        )

        if allow_instance_profile is not None:
            LOG.info(f"Using DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE: {allow_instance_profile}")
            cmds.append(
                f'echo "Environment=DEADLINE_WORKER_ALLOW_INSTANCE_PROFILE={allow_instance_profile}" >> /etc/systemd/system/deadline-worker.service.d/config.conf',
            )

        if endpoint_url_deadline is not None:
            LOG.info(f"Using AWS_ENDPOINT_URL_DEADLINE: {endpoint_url_deadline}")
            cmds.append(
                f'echo "Environment=AWS_ENDPOINT_URL_DEADLINE={endpoint_url_deadline}" >> /etc/systemd/system/deadline-worker.service.d/config.conf',
            )

        if config.no_local_session_logs:
            cmds.append(
                'echo "Environment=DEADLINE_WORKER_LOCAL_SESSION_LOGS=false" >> /etc/systemd/system/deadline-worker.service.d/config.conf',
            )

            cmds.append("systemctl daemon-reload")
        if config.worker_env_var:
            for key, value in config.worker_env_var.items():
                cmds.append(
                    f"echo 'Environment=\"{key}={value}\"' >> /etc/systemd/system/deadline-worker.service.d/config.conf",
                )
            cmds.append("systemctl daemon-reload")

        return " && ".join(cmds)


@dataclass
class PosixInstanceBuildWorker(PosixInstanceWorkerBase):
    """
    This class represents a Linux EC2 Worker Host.
    Any commands must be written in Bash.
    """

    def configure_worker_command(
        self, config: DeadlineWorkerConfiguration
    ) -> str:  # pragma: no cover
        """Get the command to configure the Worker. This must be run as root."""
        cmds = [
            "set -x",
            "source /opt/deadline/worker/bin/activate",
            f"AWS_DEFAULT_REGION={self.configuration.region}",
            config.worker_agent_install.install_command_for_linux,
            *(config.pre_install_commands or []),
            # fmt: off
            (
                "install-deadline-worker "
                + "-y "
                + f"--farm-id {config.farm_id} "
                + f"--fleet-id {config.fleet.id} "
                + f"--region {config.region} "
                + f"--user {config.agent_user} "
                + f"--group {config.job_user_group} "
                + f"{'--allow-shutdown ' if config.allow_shutdown else ''}"
                + f"{'--no-install-service ' if config.no_install_service else ''}"
                + f"{'--disallow-instance-profile ' if config.disallow_instance_profile else ''}"
                + (
                    f"--session-root-dir {config.session_root_dir} "
                    if config.session_root_dir is not None
                    else ""
                )
            ),
            # fmt: on
            f"runuser --login {self.configuration.agent_user} --command 'echo \"source /opt/deadline/worker/bin/activate\" >> $HOME/.bashrc'",
        ]

        for job_user in self.configuration.job_users:
            cmds.append(f"usermod -a -G {job_user.group} {self.configuration.agent_user}")

        sudoer_rule_users = ",".join(
            [
                self.configuration.agent_user,
                *[job_user.user for job_user in self.configuration.job_users],
            ]
        )
        cmds.append(
            f'echo "{self.configuration.agent_user} ALL=({sudoer_rule_users}) NOPASSWD: ALL" > /etc/sudoers.d/{self.configuration.agent_user}'
        )

        cmds.append(self.configure_agent_user_environment(config))

        return " && ".join(cmds)


@dataclass
class DockerContainerWorker(DeadlineWorker):
    configuration: DeadlineWorkerConfiguration

    _container_id: Optional[str] = field(init=False, default=None)

    def __post_init__(self) -> None:
        # Do not install Worker agent service since it's recommended to avoid systemd usage on Docker containers
        self.configuration = replace(self.configuration, no_install_service=True)

    def start(self) -> None:
        self._tmpdir = pathlib.Path(tempfile.mkdtemp())

        assert (
            len(self.configuration.job_users) == 1
        ), f"Multiple job users not supported on Docker worker: {self.configuration.job_users}"
        # Environment variables for "run_container.sh"
        run_container_env = {
            **os.environ,
            "FARM_ID": self.configuration.farm_id,
            "FLEET_ID": self.configuration.fleet.id,
            "AGENT_USER": self.configuration.agent_user,
            "SHARED_GROUP": self.configuration.job_user_group,
            "JOB_USER": self.configuration.job_users[0].user,
            "CONFIGURE_WORKER_AGENT_CMD": self.configure_worker_command(
                config=self.configuration,
            ),
        }

        LOG.info(f"Staging Docker build context directory {self._tmpdir!s}")
        shutil.copytree(DOCKER_CONTEXT_DIR, str(self._tmpdir), dirs_exist_ok=True)

        if self.configuration.file_mappings:
            # Stage a special dir with files to copy over to a temp folder in the Docker container
            # The container is responsible for copying files from that temp folder into the final destinations
            file_mappings_dir = self._tmpdir / "file_mappings"
            os.makedirs(str(file_mappings_dir))

            # Mapping of files in temp Docker container folder to their final destination
            docker_file_mappings: dict[str, str] = {}
            for src, dst in self.configuration.file_mappings:
                src_file_name = os.path.basename(src)

                # The Dockerfile copies the file_mappings dir in the build context to "/file_mappings" in the container
                # Build up an array of mappings from "/file_mappings" to their final destination
                src_docker_path = posixpath.join("/file_mappings", src_file_name)
                assert src_docker_path not in docker_file_mappings, (
                    "Duplicate paths generated for file mappings. All source files must have unique "
                    + f"filenames. Mapping: {self.configuration.file_mappings}"
                )
                docker_file_mappings[src_docker_path] = dst

                # Copy the file over to the stage directory
                staged_dst = str(file_mappings_dir / src_file_name)
                LOG.info(f"Copying file {src} to {staged_dst}")
                shutil.copyfile(src, staged_dst)

            run_container_env["FILE_MAPPINGS"] = json.dumps(docker_file_mappings)

        # Build and start the container
        LOG.info("Starting Docker container")
        try:
            proc = subprocess.Popen(
                args="./run_container.sh",
                cwd=str(self._tmpdir),
                env=run_container_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )

            # Live logging of Docker build
            assert proc.stdout
            with proc.stdout:
                for line in iter(proc.stdout.readline, ""):
                    LOG.info(line.rstrip("\r\n"))
        except Exception as e:  # pragma: no cover
            LOG.exception(f"Failed to start Worker agent Docker container: {e}")
            _handle_subprocess_error(e)
            raise
        else:
            exit_code = proc.wait(timeout=60)
            assert exit_code == 0, f"Process failed with exit code {exit_code}"

        # Grab the container ID from --cidfile
        try:
            self._container_id = subprocess.check_output(
                args=["cat", ".container_id"],
                cwd=str(self._tmpdir),
                text=True,
                encoding="utf-8",
                timeout=1,
            ).rstrip("\r\n")
        except Exception as e:  # pragma: no cover
            LOG.exception(f"Failed to get Docker container ID: {e}")
            _handle_subprocess_error(e)
            raise
        else:
            LOG.info(f"Started Docker container {self._container_id}")

    def stop(self) -> None:
        assert (
            self._container_id
        ), "Cannot stop Docker container: Container ID is not set. Has the Docker container been started yet?"

        LOG.info(f"Terminating Worker agent process in Docker container {self._container_id}")
        try:
            self.send_command(f"pkill --signal term -f {self.configuration.agent_user}")
        except Exception as e:  # pragma: no cover
            LOG.exception(f"Failed to terminate Worker agent process: {e}")
            raise
        else:
            LOG.info("Worker agent process terminated")

        LOG.info(f"Stopping Docker container {self._container_id}")
        try:
            subprocess.check_output(
                args=["docker", "container", "stop", self._container_id],
                cwd=str(self._tmpdir),
                text=True,
                encoding="utf-8",
                timeout=30,
            )
        except Exception as e:  # pragma: noc over
            LOG.exception(f"Failed to stop Docker container {self._container_id}: {e}")
            _handle_subprocess_error(e)
            raise
        else:
            LOG.info(f"Stopped Docker container {self._container_id}")
            self._container_id = None

    def configure_worker_command(
        self, config: DeadlineWorkerConfiguration
    ) -> str:  # pragma: no cover
        """Get the command to configure the Worker. This must be run as root."""

        return ""

    def send_command(self, command: str, *, quiet: bool = False) -> CommandResult:
        assert (
            self._container_id
        ), "Container ID not set. Has the Docker container been started yet?"

        if not quiet:  # pragma: no cover
            LOG.info(f"Sending command '{command}' to Docker container {self._container_id}")
        try:
            result = subprocess.run(
                args=[
                    "docker",
                    "exec",
                    self._container_id,
                    "/bin/bash",
                    "-euo",
                    "pipefail",
                    "-c",
                    command,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
            )
        except Exception as e:
            if not quiet:  # pragma: no cover
                LOG.exception(f"Failed to run command: {e}")
                _handle_subprocess_error(e)
            raise
        else:
            return CommandResult(
                exit_code=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )

    def get_worker_id(self) -> str:
        cmd_result: Optional[CommandResult] = None

        def got_worker_id() -> bool:
            nonlocal cmd_result
            try:
                cmd_result = self.send_command(
                    "cat /var/lib/deadline/worker.json | jq -r '.worker_id' || (cat /var/log/amazon/deadline/worker.log; false)",
                    quiet=True,
                )
            except subprocess.CalledProcessError as e:
                LOG.warning(f"Worker ID retrieval failed: {e}")
                return False
            else:
                return cmd_result.exit_code == 0

        wait_for(
            description="retrieval of worker ID from /var/lib/deadline/worker.json",
            predicate=got_worker_id,
            interval_s=10,
            max_retries=6,
        )

        assert isinstance(cmd_result, CommandResult)
        cmd_result = cast(CommandResult, cmd_result)
        assert cmd_result.exit_code == 0, f"Failed to get Worker ID: {cmd_result}"

        worker_id = cmd_result.stdout.rstrip("\r\n")
        assert re.match(
            r"^worker-[0-9a-f]{32}$", worker_id
        ), f"Got nonvalid Worker ID from command stdout: {cmd_result}"

        return worker_id

    @property
    def container_id(self) -> str | None:
        return self._container_id


def _handle_subprocess_error(e: Any) -> None:  # pragma: no cover
    if hasattr(e, "stdout"):
        LOG.error(f"Command stdout: {e.stdout}")
    if hasattr(e, "stderr"):
        LOG.error(f"Command stderr: {e.stderr}")
