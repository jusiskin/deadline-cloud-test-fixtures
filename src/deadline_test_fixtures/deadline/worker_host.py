# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
from __future__ import annotations

import abc
import botocore.client
import botocore.exceptions
import glob
import json
import logging
import os
import shlex

from dataclasses import dataclass, field, InitVar
from enum import Enum
from typing import Optional


from ..util import call_api, retry_with_predicate, is_instance_not_ready, wait_for

LOG = logging.getLogger(__name__)


@dataclass(frozen=True)
class CommandResult:  # pragma: no cover
    exit_code: int
    stdout: str
    stderr: Optional[str] = None

    def __str__(self) -> str:
        return "\n".join(
            [
                f"exit_code: {self.exit_code}",
                "",
                "================================",
                "========= BEGIN stdout =========",
                "================================",
                "",
                self.stdout,
                "",
                "==============================",
                "========= END stdout =========",
                "==============================",
                "",
                "================================",
                "========= BEGIN stderr =========",
                "================================",
                "",
                str(self.stderr),
                "",
                "==============================",
                "========= END stderr =========",
                "==============================",
            ]
        )


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
            raise WorkerHostError(
                message="Worker host is already running",
                host_os=self._operating_system(),
                diagnostics="Cannot start a host that is already in RUNNING state",
            )
        if self._state == WorkerHostState.STOPPED:
            raise WorkerHostError(
                message="Cannot restart a stopped worker host",
                host_os=self._operating_system(),
                diagnostics="Worker hosts cannot be restarted after being stopped. Create a new host instead.",
            )

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
            raise WorkerHostError(
                message="Worker host is already stopped",
                host_os=self._operating_system(),
                diagnostics="Cannot stop a host that is already in STOPPED state",
            )

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
            WorkerHostError: If another worker already has an agent on this host
        """
        if self._active_worker_id is not None and self._active_worker_id != worker_id:
            raise WorkerHostError(
                message="Cannot start worker agent: another worker already has an agent running on this host",
                host_os=self._operating_system(),
                instance_id=getattr(self, "instance_id", None),
                diagnostics=f"Active worker ID: {self._active_worker_id}. Call stop() on that worker first.",
            )
        self._active_worker_id = worker_id

    def _release_from_worker(self, worker_id: int) -> None:
        """Release this host from a worker agent."""
        if self._active_worker_id == worker_id:
            self._active_worker_id = None


@dataclass
class Ec2Tag:
    """EC2 instance tag."""

    key: str
    value: str


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


DEFAULT_WAITER_CONFIG = {
    "Delay": 5,
    "MaxAttempts": 30,
}


@dataclass
class EC2WorkerHost(WorkerHost):
    """Base class for EC2 worker hosts."""

    subnet_id: str
    security_group_id: str
    instance_profile_name: str
    bootstrap_bucket_name: str
    s3_client: botocore.client.BaseClient
    ec2_client: botocore.client.BaseClient
    ssm_client: botocore.client.BaseClient
    instance_type: str
    instance_shutdown_behavior: str
    additional_tags: list[Ec2Tag] = field(default_factory=list)
    instance_id: Optional[str] = field(init=False, default=None)
    override_ami_id: InitVar[Optional[str]] = None

    # Userdata success/failure constants
    USERDATA_SUCCESS_STRING: str = field(default="Userdata finished successfully", init=False)
    USERDATA_FAILURE_STRING: str = field(default="Userdata failed to finish", init=False)

    def __post_init__(self, override_ami_id: Optional[str] = None):
        super().__init__()
        if override_ami_id:
            self._ami_id = override_ami_id

    @abc.abstractmethod
    def ami_ssm_param_name(self) -> str:
        """Return the SSM parameter name for the AMI."""
        pass

    @abc.abstractmethod
    def ssm_document_name(self) -> str:
        """Return the SSM document name for sending commands."""
        pass

    @abc.abstractmethod
    def userdata(self, s3_files: list[tuple[str, str]] | None) -> str:
        """Generate userdata script for instance launch."""
        pass

    @abc.abstractmethod
    def ebs_devices(self) -> dict[str, int] | None:
        """Return EBS device mappings."""
        pass

    @abc.abstractmethod
    def userdata_success_script(self) -> str:
        """Generate script to check if userdata finished successfully."""
        pass

    @property
    def ami_id(self) -> str:
        """Get the AMI ID, resolving from SSM parameter if needed."""
        if not hasattr(self, "_ami_id"):
            response = call_api(
                description=f"Getting latest {type(self)} AMI ID from SSM parameter {self.ami_ssm_param_name()}",
                fn=lambda: self.ssm_client.get_parameters(Names=[self.ami_ssm_param_name()]),
            )

            parameters = response.get("Parameters", [])
            assert (
                len(parameters) == 1
            ), f"Received incorrect number of SSM parameters. Expected 1, got response: {response}"
            self._ami_id = parameters[0]["Value"]
            LOG.info(f"Using latest {type(self)} AMI {self._ami_id}")

        return self._ami_id

    def _do_start(self) -> None:
        """Start the EC2 instance and wait for userdata to complete."""
        self._launch_instance()
        # Temporarily set state to RUNNING so send_command works during userdata check
        self._state = WorkerHostState.RUNNING
        try:
            self._wait_until_userdata_finishes()
        except Exception:
            # If userdata fails, reset state to NOT_STARTED
            self._state = WorkerHostState.NOT_STARTED
            raise

    def _wait_until_userdata_finishes(self) -> None:
        """Wait for userdata to complete successfully."""
        if not self.instance_id:
            raise WorkerHostError(
                message="Cannot wait for userdata: no instance ID available",
                host_os=self._operating_system(),
                diagnostics="Instance ID is None. The instance may not have been launched successfully.",
            )

        result: Optional[CommandResult] = None
        success: bool = False
        LOG.info("Waiting for userdata to finish")

        def get_userdata_result() -> bool:
            nonlocal result
            nonlocal success
            result = self.send_command(self.userdata_success_script())

            if self.USERDATA_SUCCESS_STRING in str(result):
                success = True
                return True

            if self.USERDATA_FAILURE_STRING in str(result):
                success = False
                return True

            return False

        try:
            # Raises TimeoutError if the userdata status cannot be fetched in
            # the given timeframe.
            wait_for(
                description="getting the result of userdata",
                predicate=get_userdata_result,
                interval_s=5,
                max_retries=60,
            )
        except TimeoutError as e:
            raise WorkerHostError(
                message=f"Timeout waiting for userdata to complete on instance {self.instance_id}",
                host_os=self._operating_system(),
                instance_id=self.instance_id,
                diagnostics="Userdata did not complete within 300 seconds (60 retries × 5s intervals)",
            ) from e

        if not success:
            # Userdata failed - include the failure details in the error
            failure_details = str(result) if result else "No result available"
            raise WorkerHostError(
                message=f"Userdata failed on instance {self.instance_id}",
                host_os=self._operating_system(),
                instance_id=self.instance_id,
                diagnostics=f"Userdata failure details:\n{failure_details}",
            )

        LOG.info("Userdata finished successfully.")

    @retry_with_predicate(
        max_attempts=3, predicate=lambda e: isinstance(e, botocore.exceptions.WaiterError)
    )
    @retry_with_predicate(max_attempts=60, delay=10, backoff=1, predicate=is_instance_not_ready)
    def _send_command_internal(
        self, command: str, ssm_waiter_config: dict[str, int] = DEFAULT_WAITER_CONFIG
    ) -> CommandResult:
        """Send a command via SSM without checking if host is running (for internal use during startup)."""
        if not self.instance_id:
            raise WorkerHostError(
                message="No instance ID available",
                host_os=self._operating_system(),
                diagnostics="Cannot send command without an instance ID. The instance may not have been launched.",
            )

        ssm_waiter = self.ssm_client.get_waiter("command_executed")

        LOG.info(f"Sending SSM command to instance {self.instance_id}")
        try:
            send_command_response = self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName=self.ssm_document_name(),
                Parameters={"commands": [command]},
            )
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "InvalidInstanceId":
                LOG.warning(
                    f"Instance {self.instance_id} is not ready for SSM command (received InvalidInstanceId error)."
                )
            raise

        command_id = send_command_response["Command"]["CommandId"]

        LOG.info(f"Waiting for SSM command {command_id} to reach a terminal state")
        try:
            ssm_waiter.wait(
                InstanceId=self.instance_id,
                CommandId=command_id,
                WaiterConfig=ssm_waiter_config,
            )
        except botocore.exceptions.WaiterError as e:
            LOG.warning(f"WaiterError caught for command {command_id}:")
            LOG.warning(f"\tError reason: {str(e)}")
            LOG.warning(f"\tWaiter last response: {str(e.last_response)}")

            if isinstance(e, botocore.exceptions.WaiterError) and (
                "Undeliverable" in str(e) or "Undeliverable" in str(e.last_response)
            ):
                LOG.warning(
                    f"Unable to deliver command {command_id} to instance {self.instance_id} (received UndeliverableError)."
                )
                raise e

        ssm_command_result = self.ssm_client.get_command_invocation(
            InstanceId=self.instance_id,
            CommandId=command_id,
        )
        result = CommandResult(
            exit_code=ssm_command_result["ResponseCode"],
            stdout=ssm_command_result["StandardOutputContent"],
            stderr=ssm_command_result["StandardErrorContent"],
        )
        if result.exit_code == -1:
            LOG.error(f"Failed to send SSM command {command_id} to {self.instance_id}: {result}")

        LOG.info(f"SSM command {command_id} completed with exit code: {result.exit_code}")
        return result

    def _do_stop(self) -> None:
        """Stop the EC2 instance and clean up resources."""
        if self.instance_id:
            LOG.info(f"Terminating EC2 instance {self.instance_id}")
            try:
                self.ec2_client.terminate_instances(InstanceIds=[self.instance_id])
            except Exception as e:
                LOG.warning(f"Failed to terminate instance {self.instance_id}: {e}")
            finally:
                self.instance_id = None

    @retry_with_predicate(
        max_attempts=3, predicate=lambda e: isinstance(e, botocore.exceptions.WaiterError)
    )
    @retry_with_predicate(max_attempts=60, delay=10, backoff=1, predicate=is_instance_not_ready)
    def send_command(
        self, command: str, ssm_waiter_config: dict[str, int] = DEFAULT_WAITER_CONFIG
    ) -> CommandResult:
        """Send a command via SSM to a shell on a launched EC2 instance."""
        if not self.is_running():
            raise WorkerHostError(
                message="Cannot send command to non-running host",
                host_os=self._operating_system(),
                instance_id=getattr(self, "instance_id", None),
                diagnostics=f"Host state is {self._state}. Host must be in RUNNING state to send commands.",
            )

        if not self.instance_id:
            raise WorkerHostError(
                message="No instance ID available",
                host_os=self._operating_system(),
                diagnostics="Cannot send command without an instance ID.",
            )

        ssm_waiter = self.ssm_client.get_waiter("command_executed")

        LOG.info(f"Sending SSM command to instance {self.instance_id}")
        try:
            send_command_response = self.ssm_client.send_command(
                InstanceIds=[self.instance_id],
                DocumentName=self.ssm_document_name(),
                Parameters={"commands": [command]},
            )
        except botocore.exceptions.ClientError as error:
            if error.response["Error"]["Code"] == "InvalidInstanceId":
                LOG.warning(
                    f"Instance {self.instance_id} is not ready for SSM command (received InvalidInstanceId error)."
                )
            raise

        command_id = send_command_response["Command"]["CommandId"]

        LOG.info(f"Waiting for SSM command {command_id} to reach a terminal state")
        try:
            ssm_waiter.wait(
                InstanceId=self.instance_id,
                CommandId=command_id,
                WaiterConfig=ssm_waiter_config,
            )
        except botocore.exceptions.WaiterError as e:
            LOG.warning(f"WaiterError caught for command {command_id}:")
            LOG.warning(f"\tError reason: {str(e)}")
            LOG.warning(f"\tWaiter last response: {str(e.last_response)}")

            if isinstance(e, botocore.exceptions.WaiterError) and (
                "Undeliverable" in str(e) or "Undeliverable" in str(e.last_response)
            ):
                LOG.warning(
                    f"Unable to deliver command {command_id} to instance {self.instance_id} (received UndeliverableError)."
                )
                raise e

        ssm_command_result = self.ssm_client.get_command_invocation(
            InstanceId=self.instance_id,
            CommandId=command_id,
        )
        result = CommandResult(
            exit_code=ssm_command_result["ResponseCode"],
            stdout=ssm_command_result["StandardOutputContent"],
            stderr=ssm_command_result["StandardErrorContent"],
        )
        if result.exit_code == -1:
            LOG.error(f"Failed to send SSM command {command_id} to {self.instance_id}: {result}")

        LOG.info(f"SSM command {command_id} completed with exit code: {result.exit_code}")
        return result

    def _launch_instance(self, *, s3_files: list[tuple[str, str]] | None = None) -> None:
        """Launch the EC2 instance."""
        assert (
            not self.instance_id
        ), "Attempted to launch EC2 instance when one was already launched"
        try:
            LOG.info("Launching EC2 instance")
            LOG.info(
                json.dumps(
                    {
                        "AMI_ID": self.ami_id,
                        "Instance Profile": self.instance_profile_name,
                        "User Data": self.userdata(s3_files),
                    },
                    indent=4,
                    sort_keys=True,
                )
            )

            tags = [
                {
                    "Key": "InstanceIdentification",
                    "Value": "DeadlineScaffoldingWorker",
                }
            ]

            for tag in self.additional_tags:
                tags.append({"Key": tag.key, "Value": tag.value})

            run_instance_request = {
                "MinCount": 1,
                "MaxCount": 1,
                "ImageId": self.ami_id,
                "InstanceType": self.instance_type,
                "IamInstanceProfile": {"Name": self.instance_profile_name},
                "SubnetId": self.subnet_id,
                "SecurityGroupIds": [self.security_group_id],
                "MetadataOptions": {"HttpTokens": "required", "HttpEndpoint": "enabled"},
                "TagSpecifications": [
                    {
                        "ResourceType": "instance",
                        "Tags": tags,
                    }
                ],
                "InstanceInitiatedShutdownBehavior": self.instance_shutdown_behavior,
                "UserData": self.userdata(s3_files),
            }

            devices = self.ebs_devices() or {}
            device_mappings = [
                {"DeviceName": name, "Ebs": {"VolumeSize": size}} for name, size in devices.items()
            ]
            if device_mappings:
                run_instance_request["BlockDeviceMappings"] = device_mappings

            run_instance_response = self.ec2_client.run_instances(**run_instance_request)

            self.instance_id = run_instance_response["Instances"][0]["InstanceId"]
            LOG.info(f"Launched EC2 instance {self.instance_id}")

            LOG.info(f"Waiting for EC2 instance {self.instance_id} status to be OK")
            instance_running_waiter = self.ec2_client.get_waiter("instance_status_ok")
            instance_running_waiter.wait(
                InstanceIds=[self.instance_id],
                WaiterConfig={"Delay": 15, "MaxAttempts": 75},
            )
            LOG.info(f"EC2 instance {self.instance_id} status is OK")
        except botocore.exceptions.WaiterError as e:
            diagnostics = self._collect_instance_diagnostics()
            raise InstanceStartupError(
                message=f"Failed to wait for instance status: {e}", diagnostics=diagnostics
            ) from e
        except Exception as e:
            LOG.error(f"Unexpected error during instance launch: {e}")
            raise

    def _collect_instance_diagnostics(self) -> str:
        """Collect diagnostic information about the instance."""
        if not self.instance_id:
            return "No instance_id available for diagnostics"

        diagnostic_info = []
        diagnostic_info.append(f"Collecting diagnostics for instance {self.instance_id}")

        # Get instance details
        try:
            instance_response = self.ec2_client.describe_instances(InstanceIds=[self.instance_id])
            instance = instance_response["Reservations"][0]["Instances"][0]

            # Log instance details
            diagnostic_info.append(f"Instance state: {instance['State']['Name']}")
            diagnostic_info.append(f"Instance type: {instance['InstanceType']}")
            diagnostic_info.append(f"Launch time: {instance['LaunchTime']}")
            diagnostic_info.append(
                f"Availability zone: {instance['Placement']['AvailabilityZone']}"
            )
        except Exception as e:
            diagnostic_info.append(f"Failed to get instance details: {e}")

        # Get instance status
        try:
            status_response = self.ec2_client.describe_instance_status(
                InstanceIds=[self.instance_id], IncludeAllInstances=True
            )
            if status_response["InstanceStatuses"]:
                status = status_response["InstanceStatuses"][0]
                diagnostic_info.append(
                    f"System status: {status.get('SystemStatus', {}).get('Status', 'unknown')}"
                )
                diagnostic_info.append(
                    f"Instance status: {status.get('InstanceStatus', {}).get('Status', 'unknown')}"
                )

                # Log status check details if available
                if "SystemStatus" in status and "Details" in status["SystemStatus"]:
                    for detail in status["SystemStatus"]["Details"]:
                        diagnostic_info.append(
                            f"System check {detail.get('Name')}: {detail.get('Status')}"
                        )

                if "InstanceStatus" in status and "Details" in status["InstanceStatus"]:
                    for detail in status["InstanceStatus"]["Details"]:
                        diagnostic_info.append(
                            f"Instance check {detail.get('Name')}: {detail.get('Status')}"
                        )
        except Exception as e:
            diagnostic_info.append(f"Failed to get instance status: {e}")
        return "\n".join(diagnostic_info)

    def transfer_files(
        self, file_mappings: list[tuple[str, str]], s3_key_prefix: str = "worker"
    ) -> None:
        """
        Transfer files from local machine to the EC2 instance.

        This method handles the complete file transfer process:
        1. Uploads files from local machine to S3 (staging)
        2. Uses SSM to download files from S3 to the EC2 instance

        Args:
            file_mappings: List of (source_glob, destination_path) tuples where:
                - source_glob: Local file path or glob pattern
                - destination_path: Destination path on the EC2 instance
            s3_key_prefix: Prefix for S3 keys (default: "worker")

        Raises:
            AssertionError: If duplicate S3 keys would be generated
            botocore.exceptions.ClientError: If S3 upload fails
            RuntimeError: If file download to EC2 instance fails
        """
        if not file_mappings:
            LOG.info("No file mappings to transfer")
            return

        # Step 1: Upload files to S3
        s3_to_src_mapping: dict[str, str] = {}
        s3_to_dst_mapping: dict[str, str] = {}

        for src_glob, dst in file_mappings:
            for src_file in glob.glob(src_glob):
                s3_key = f"{s3_key_prefix}/{os.path.basename(src_file)}"
                assert s3_key not in s3_to_src_mapping, (
                    "Duplicate S3 keys generated for file mappings. All source files must have unique "
                    + f"filenames. Mapping: {file_mappings}"
                )
                s3_to_src_mapping[s3_key] = src_file
                s3_to_dst_mapping[f"s3://{self.bootstrap_bucket_name}/{s3_key}"] = dst

        for key, local_path in s3_to_src_mapping.items():
            LOG.info(f"Uploading file {local_path} to s3://{self.bootstrap_bucket_name}/{key}")
            try:
                with open(local_path, mode="rb") as f:
                    self.s3_client.put_object(
                        Bucket=self.bootstrap_bucket_name,
                        Key=key,
                        Body=f,
                    )
            except botocore.exceptions.ClientError as e:
                LOG.exception(
                    f"Failed to upload file {local_path} to s3://{self.bootstrap_bucket_name}/{key}: {e}"
                )
                raise

        # Step 2: Download files from S3 to EC2 instance
        s3_to_dst_list = list(s3_to_dst_mapping.items())
        if s3_to_dst_list:
            LOG.info(f"Transferring {len(s3_to_dst_list)} files from S3 to EC2 instance")
            download_command = self._get_download_files_command(s3_to_dst_list)
            result = self.send_command(download_command)
            if result.exit_code != 0:
                raise RuntimeError(f"Failed to download files from S3 to EC2 instance: {result}")
            LOG.info(f"Successfully transferred {len(s3_to_dst_list)} files to EC2 instance")

    @abc.abstractmethod
    def _get_download_files_command(self, s3_files: list[tuple[str, str]]) -> str:
        """Get the OS-specific command to download files from S3 to the instance."""
        pass

    @abc.abstractmethod
    def _get_remove_files_command(self, file_paths: list[str]) -> str:
        """Get the OS-specific command to remove multiple files in a single command."""
        pass

    def cleanup_files(self, file_paths: list[str]) -> None:
        """
        Clean up files from the worker host.

        Args:
            file_paths: List of file paths to remove from the host

        Note:
            This method uses a single SSM command to remove all files for efficiency.
            Failures are logged as warnings but do not raise exceptions.
        """
        if not file_paths:
            LOG.info("No files to clean up")
            return

        LOG.info(f"Cleaning up {len(file_paths)} files from worker host")
        try:
            cleanup_command = self._get_remove_files_command(file_paths)
            self.send_command(cleanup_command)
            LOG.info(f"Successfully removed {len(file_paths)} files")
        except Exception as e:
            LOG.warning(f"Failed to remove files: {e}")


@dataclass
class WindowsEC2WorkerHost(EC2WorkerHost):
    """Windows-specific EC2 worker host."""

    WIN2022_AMI_NAME: str = field(default="Windows_Server-2022-English-Full-Base", init=False)

    # Windows-specific userdata signaling paths
    SIGNAL_USER_DATA_DIR: str = field(default="C:\\signal_user_data_finished", init=False)
    SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME: str = field(init=False)
    SIGNAL_USER_DATA_FAILED_FILE_NAME: str = field(init=False)

    def __post_init__(self, override_ami_id: Optional[str] = None):
        super().__post_init__(override_ami_id)
        # Initialize signal file paths after SIGNAL_USER_DATA_DIR is set
        self.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME = f"{self.SIGNAL_USER_DATA_DIR}\\success"
        self.SIGNAL_USER_DATA_FAILED_FILE_NAME = f"{self.SIGNAL_USER_DATA_DIR}\\failed"

    def _operating_system(self) -> str:
        return "windows"

    def ami_ssm_param_name(self) -> str:
        """Return the SSM parameter name for the Windows AMI."""
        return f"/aws/service/ami-windows-latest/{self.WIN2022_AMI_NAME}"

    def ssm_document_name(self) -> str:
        return "AWS-RunPowerShellScript"

    def ebs_devices(self) -> dict[str, int] | None:
        """DeviceName -> VolumeSize (in GiBs) mapping"""
        # defaults to 60GB to match SMF, aws gives 30GB by default
        return {"/dev/sda1": 60}

    def userdata_success_script(self) -> str:
        """Generate PowerShell script to check userdata completion status."""
        return f"""
$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest
Set-PSDebug -Trace 1
if (Test-Path "{self.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME}") {{
    echo "{self.USERDATA_SUCCESS_STRING}"
    exit 0
}}
if (Test-Path "{self.SIGNAL_USER_DATA_FAILED_FILE_NAME}") {{
    echo "{self.USERDATA_FAILURE_STRING}"
    cat "{self.SIGNAL_USER_DATA_FAILED_FILE_NAME}"
    exit 0
}}
"""

    def userdata(self, s3_files: list[tuple[str, str]] | None) -> str:
        """Generate Windows userdata script for instance launch."""
        copy_s3_command = ""

        if s3_files:
            copy_s3_command = " ; ".join([f"aws s3 cp {s3_uri} {dst}" for s3_uri, dst in s3_files])

        userdata = f"""<powershell>
try {{
    $ProgressPreference = 'SilentlyContinue'
    
    # Create signal directory
    New-Item -ItemType Directory -Force -Path "{self.SIGNAL_USER_DATA_DIR}"
    
    Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe" -OutFile "C:\\python-3.12.10-amd64.exe"
    $installerHash=(Get-FileHash "C:\\python-3.12.10-amd64.exe" -Algorithm "MD5")
    $expectedHash="5eddb0b6f12c852725de071ae681dde4"
    if ($installerHash.Hash -ne $expectedHash) {{ throw "Could not verify Python installer." }}
    Start-Process -FilePath "C:\\python-3.12.10-amd64.exe" -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 AppendPath=1" -Wait
    Invoke-WebRequest -Uri "https://awscli.amazonaws.com/AWSCLIV2.msi" -Outfile "C:\\AWSCLIV2.msi"
    Start-Process msiexec.exe -ArgumentList "/i C:\\AWSCLIV2.msi /quiet" -Wait
    $env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine")
    {copy_s3_command}
    
    # Signal success
    "Userdata completed successfully" | Out-File -FilePath "{self.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME}" -Encoding UTF8
}} catch {{
    # Signal failure with error details
    $errorMessage = "Userdata failed: $($_.Exception.Message)`nStack trace: $($_.ScriptStackTrace)"
    $errorMessage | Out-File -FilePath "{self.SIGNAL_USER_DATA_FAILED_FILE_NAME}" -Encoding UTF8
    throw
}}
</powershell>"""

        return userdata

    def _get_download_files_command(self, s3_files: list[tuple[str, str]]) -> str:
        """Get the Windows PowerShell command to download files from S3."""
        # Build a PowerShell command that downloads all files from S3
        download_commands = [f"aws s3 cp {s3_uri} {dst}" for s3_uri, dst in s3_files]
        return " ; ".join(download_commands)

    def _get_remove_files_command(self, file_paths: list[str]) -> str:
        """Get the Windows PowerShell command to remove multiple files in a single command."""
        # Build a PowerShell command that removes all files
        # Use ForEach-Object to iterate through the paths and remove each one
        paths_array = ", ".join([f'"{path}"' for path in file_paths])
        return f"@({paths_array}) | ForEach-Object {{ Remove-Item -Path $_ -Force -ErrorAction SilentlyContinue }}"


@dataclass
class PosixEC2WorkerHost(EC2WorkerHost):
    """POSIX (Linux)-specific EC2 worker host."""

    AL2023_AMI_NAME: str = field(default="al2023-ami-kernel-6.1-x86_64", init=False)

    # POSIX-specific userdata signaling paths
    SIGNAL_USER_DATA_SUCCESS_DIR: str = field(
        default="/var/tmp/signal_user_data_finished", init=False
    )
    SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME: str = field(init=False)
    SIGNAL_USER_DATA_FAILED_FILE_NAME: str = field(init=False)

    def __post_init__(self, override_ami_id: Optional[str] = None):
        super().__post_init__(override_ami_id)
        # Initialize signal file paths after SIGNAL_USER_DATA_SUCCESS_DIR is set
        self.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME = f"{self.SIGNAL_USER_DATA_SUCCESS_DIR}/success"
        self.SIGNAL_USER_DATA_FAILED_FILE_NAME = f"{self.SIGNAL_USER_DATA_SUCCESS_DIR}/failed"

    def _operating_system(self) -> str:
        return "posix"

    def ami_ssm_param_name(self) -> str:
        """Return the SSM parameter name for the POSIX AMI."""
        return f"/aws/service/ami-amazon-linux-latest/{self.AL2023_AMI_NAME}"

    def ssm_document_name(self) -> str:
        return "AWS-RunShellScript"

    def ebs_devices(self) -> dict[str, int] | None:
        """DeviceName -> VolumeSize (in GiBs) mapping"""
        # defaults to 30GB to match SMF, aws gives 8GB by default
        return {"/dev/xvda": 30}

    def send_command(
        self, command: str, ssm_waiter_config: dict[str, int] = DEFAULT_WAITER_CONFIG
    ) -> CommandResult:
        """Send a command to the POSIX host, prepending bash safety flags."""
        return super().send_command("set -euxo pipefail; " + command, ssm_waiter_config)

    def _send_command_internal(
        self, command: str, ssm_waiter_config: dict[str, int] = DEFAULT_WAITER_CONFIG
    ) -> CommandResult:
        """Send a command to the POSIX host internally, prepending bash safety flags."""
        return super()._send_command_internal("set -euxo pipefail; " + command, ssm_waiter_config)

    def userdata_success_script(self) -> str:
        """Generate bash script to check userdata completion status."""
        return f"""
if [[ -f "{self.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME}" ]]; then
    echo "{self.USERDATA_SUCCESS_STRING}"
    exit 0
fi
if [[ -f "{self.SIGNAL_USER_DATA_FAILED_FILE_NAME}" ]]; then
    echo "{self.USERDATA_FAILURE_STRING}"
    cat "{self.SIGNAL_USER_DATA_FAILED_FILE_NAME}"
    exit 0
fi
"""

    def userdata(self, s3_files: list[tuple[str, str]] | None) -> str:
        """Generate POSIX userdata script for instance launch."""
        copy_s3_command = ""

        if s3_files:
            copy_s3_command = " && ".join(
                [f"aws s3 cp {s3_uri} {dst} && chmod o+rx {dst}" for s3_uri, dst in s3_files]
            )

        userdata = f"""#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
set -x

# Create signal directory
mkdir -p "{self.SIGNAL_USER_DATA_SUCCESS_DIR}"

# Trap to signal failure on any error
trap 'echo "Userdata failed at line $LINENO: $BASH_COMMAND" > "{self.SIGNAL_USER_DATA_FAILED_FILE_NAME}"; exit 1' ERR

{copy_s3_command}

mkdir /opt/deadline
python3 -m venv /opt/deadline/worker

# Signal success
echo "Userdata completed successfully" > "{self.SIGNAL_USER_DATA_SUCCESSFUL_FILE_NAME}"
"""

        return userdata

    def _get_download_files_command(self, s3_files: list[tuple[str, str]]) -> str:
        """Get the POSIX shell command to download files from S3."""
        # Build a shell command that downloads all files from S3
        # Use && to chain commands so we fail fast if any download fails
        download_commands = [
            f"aws s3 cp {shlex.quote(s3_uri)} {shlex.quote(dst)} && chmod o+rx {shlex.quote(dst)}"
            for s3_uri, dst in s3_files
        ]
        return " && ".join(download_commands)

    def _get_remove_files_command(self, file_paths: list[str]) -> str:
        """Get the POSIX shell command to remove multiple files in a single command."""
        # Use shlex.join to properly quote file paths for shell safety
        return f"rm -f {shlex.join(file_paths)}"
