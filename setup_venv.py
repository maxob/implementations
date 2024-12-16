#!/usr/bin/env python3
"""Cross-platform virtual environment setup script."""

import argparse
import importlib.util
import logging
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def setup_logger(verbose: bool = False) -> None:
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def is_windows() -> bool:
    """Check if running on Windows."""
    return platform.system().lower() == "windows"


def check_venv_module() -> None:
    """Check if venv module is available and provide installation instructions if not."""
    if importlib.util.find_spec("venv") is None:
        msg = (
            "Python venv module not found. Please install it first:\n"
            "Ubuntu/Debian: sudo apt-get install python3-venv\n"
            "Fedora: sudo dnf install python3-venv\n"
            "Windows: Python venv should be included by default"
        )
        LOGGER.error(msg)
        raise RuntimeError(msg)


def create_venv(venv_path: Path) -> None:
    """Create a virtual environment.

    Args:
        venv_path: Path to create virtual environment in.

    """
    venv_path = venv_path.resolve()
    LOGGER.info(f"Creating virtual environment at {venv_path}")

    # Use get_system_python() instead of hardcoded paths
    system_python = get_system_python()

    try:
        # Remove target directory if it exists
        if venv_path.exists():
            import shutil

            shutil.rmtree(venv_path)
            LOGGER.debug(f"Removed existing directory at {venv_path}")

        # Create venv using explicit system Python
        result = subprocess.run(
            [system_python, "-m", "venv", str(venv_path)],
            capture_output=True,
            text=True,
            check=False,
        )

        if result.returncode != 0:
            _handle_process_error(result, "venv creation")

        LOGGER.info(f"Successfully created virtual environment using Python: {system_python}")
        LOGGER.debug(f"Virtual environment path: {venv_path}")
    except Exception:
        LOGGER.exception("Failed to create virtual environment.")
        raise


def _handle_process_error(process: subprocess.CompletedProcess, operation: str) -> None:
    """Handle process errors and raise appropriate exceptions.

    Args:
        process: Process to handle
        operation: Operation that failed

    """
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, operation)


def get_venv_python(venv_path: Path) -> str:
    """Get path to Python executable in venv.

    Args:
        venv_path: Path to virtual environment

    Returns:
        Path to Python executable

    """
    venv_path = venv_path.resolve()
    python_path = (
        venv_path
        / ("Scripts" if is_windows() else "bin")
        / ("python.exe" if is_windows() else "python")
    )
    if not python_path.exists():
        msg = f"Python executable not found at {python_path}"
        LOGGER.error(msg)
        raise FileNotFoundError(msg)
    LOGGER.debug(f"Using Python from virtual environment: {python_path}")
    return str(python_path)


def find_repo_root() -> Path:
    """Find repository root.

    Returns:
        Path to repository root

    """
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    msg = "Could not find repository root"
    raise RuntimeError(msg)


def run_with_venv(cmd: list[str], venv_path: Path) -> None:
    """Run command with virtual environment activated.

    Args:
        cmd: Command to run
        venv_path: Path to virtual environment

    """
    venv_path = venv_path.resolve()
    env = os.environ.copy()

    if is_windows():
        python_path = str(venv_path / "Scripts")
    else:
        python_path = str(venv_path / "bin")

    # Modify environment to activate venv
    env["VIRTUAL_ENV"] = str(venv_path)
    env["PATH"] = f"{python_path}{os.pathsep}{env['PATH']}"

    LOGGER.debug(f"Running command in venv: {' '.join(cmd)}")
    LOGGER.debug(f"Virtual env path: {python_path}")

    # Run process with real-time output
    process = subprocess.run(
        cmd,
        env=env,
        stdout=None,  # Don't capture, let it go to terminal
        stderr=None,  # Don't capture, let it go to terminal
        text=True,
        check=False,
    )

    if process.returncode != 0:
        _handle_process_error(process, " ".join(cmd))


def is_valid_venv(venv_path: Path) -> bool:
    """Check if virtual environment is valid and has Python executable.

    Args:
        venv_path: Path to virtual environment

    Returns:
        True if virtual environment has Python executable, False otherwise.

    """
    python_path = (
        venv_path.resolve()
        / ("Scripts" if is_windows() else "bin")
        / ("python.exe" if is_windows() else "python")
    )
    return python_path.exists()


def setup_venv(venv_path: Path, force: bool = False, verbose: bool = False) -> None:
    """Set up virtual environment and install packages.

    Args:
        venv_path: Path to create virtual environment
        force: Force recreation of virtual environment
        verbose: Enable verbose output

    """
    LOGGER.info("Starting virtual environment setup...")
    LOGGER.debug(f"Current working directory: {Path.cwd().resolve()}")
    LOGGER.debug(f"Force reinstall: {force}")
    LOGGER.debug(f"Verbose mode: {verbose}")

    venv_path = venv_path.resolve()

    # Check existing venv
    if venv_path.exists():
        if force or not is_valid_venv(venv_path):
            if force:
                LOGGER.info(f"Force flag set. Removing existing virtual environment at {venv_path}")
            else:
                LOGGER.info(f"Invalid virtual environment found at {venv_path}. Recreating...")
            import shutil

            shutil.rmtree(venv_path)
            LOGGER.debug("Successfully removed existing virtual environment")
        else:
            LOGGER.info(f"Using existing valid virtual environment at {venv_path}")

    # Create venv if needed
    if not venv_path.exists():
        create_venv(venv_path)
        LOGGER.debug("Verifying virtual environment...")
        if not is_valid_venv(venv_path):
            msg = "Failed to create a valid virtual environment"
            LOGGER.error(msg)
            raise RuntimeError(msg)

    # Note(max): for some reason this does't work on older Windows - fix?
    # Upgrade pip for the new environment
    # LOGGER.debug("Upgrading pip for the new environment...")
    # run_with_venv(["-m", "pip", "install", "--upgrade", "pip"], venv_path)
    # LOGGER.debug("Pip upgraded successfully")

    # Find repository root
    repo_root = find_repo_root()
    LOGGER.info(f"Found repository root at {repo_root}")

    # Import and run install_requirements
    try:
        install_req_script = repo_root / "install_requirements.py"
        LOGGER.info(f"Using install_requirements.py from: {install_req_script}")

        cmd = [
            str(get_venv_python(venv_path)),
            str(install_req_script),
            *(["--verbose"] if verbose else []),
            *(["--force"] if force else []),
        ]

        LOGGER.info("Starting package installation using install_requirements.py")
        run_with_venv(cmd, venv_path)
        LOGGER.info("Package installation completed")

        # Log final activation command for user
        if is_windows():
            activate_cmd = f"{venv_path}\\Scripts\\activate.bat"
        else:
            activate_cmd = f"source {venv_path}/bin/activate"
        LOGGER.info(f"\nTo activate the virtual environment, run:\n{activate_cmd}")

    except subprocess.CalledProcessError as e:
        msg = "Package installation failed"
        LOGGER.exception(msg)
        if e.stderr:
            LOGGER.exception(e.stderr)
        raise


def get_system_python() -> str:
    """Get system Python path with better Windows support."""
    if is_windows():
        # Check common Windows Python locations
        possible_paths = [
            Path(sys.prefix) / "python.exe",
            Path("python.exe"),  # If in PATH
            Path(
                os.environ.get("LOCALAPPDATA", ""), "Programs", "Python", "Python3*", "python.exe"
            ),
        ]
        for path in possible_paths:
            if path.exists():
                return str(path)
        msg = "Could not find system Python on Windows"
        LOGGER.error(msg)
        raise FileNotFoundError(msg)
    return "/usr/bin/python3"


def deactivate_virtual_env() -> None:
    """Deactivate current virtual environment if active."""
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        LOGGER.info("Deactivating current virtual environment...")
        # Reset PATH
        original_path = os.environ["PATH"].split(os.pathsep)
        venv_path = Path(sys.executable).parent
        os.environ["PATH"] = os.pathsep.join(
            [p for p in original_path if not Path(p).is_relative_to(venv_path)]
        )
        # Reset env variables
        os.environ.pop("VIRTUAL_ENV", None)
        sys.prefix = sys.base_prefix
        LOGGER.info("Virtual environment deactivated")


def is_active_venv() -> bool:
    """Check if running in a virtual environment."""
    return hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )


@dataclass
class EnvState:
    """Store environment state."""

    path: str
    virtual_env: str | None
    python_path: str
    sys_prefix: str


def get_env_state() -> EnvState:
    """Capture current environment state."""
    return EnvState(
        path=os.environ.get("PATH", ""),
        virtual_env=os.environ.get("VIRTUAL_ENV"),
        python_path=sys.executable,
        sys_prefix=sys.prefix,
    )


def restore_env_state(state: EnvState) -> None:
    """Restore environment to previous state."""
    os.environ["PATH"] = state.path
    if state.virtual_env:
        os.environ["VIRTUAL_ENV"] = state.virtual_env
    sys.prefix = state.sys_prefix


def safe_deactivate_virtual_env() -> None:
    """Safely deactivate virtual environment with error handling."""
    if not is_active_venv():
        LOGGER.debug("No active virtual environment detected")
        return

    try:
        original_state = get_env_state()
        deactivate_virtual_env()
        LOGGER.debug("Virtual environment successfully deactivated")
    except Exception as e:
        msg = f"Failed to deactivate virtual environment: {e}"
        LOGGER.exception(msg)
        restore_env_state(original_state)
        raise RuntimeError(msg) from e


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Set up virtual environment")
    parser.add_argument(
        "--venv-path",
        type=Path,
        default=Path(".venv"),
        help="Path to create virtual environment",
    )
    parser.add_argument(
        "-f", "--force", action="store_true", help="Force recreation of virtual environment"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    return parser.parse_args()


def main() -> None:
    """Set up virtual environment."""
    args = parse_args()
    setup_logger(args.verbose)

    try:
        safe_deactivate_virtual_env()
        LOGGER.info("Starting setup_venv.py")
        LOGGER.debug(f"Python version: {sys.version}")
        LOGGER.debug(f"Platform: {platform.platform()}")
        LOGGER.debug(f"Arguments: {args}")

        # Check for venv module before proceeding
        check_venv_module()
        setup_venv(args.venv_path, args.force, args.verbose)
        LOGGER.info("Virtual environment setup completed successfully")
    except Exception:
        LOGGER.exception("Setup failed.")
        if args.verbose:
            LOGGER.exception("Detailed error information:")
        sys.exit(1)


if __name__ == "__main__":
    main()
