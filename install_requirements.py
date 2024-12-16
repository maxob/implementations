#!/usr/bin/env python3
"""Install Python packages using top-level requirements as constraints."""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

LOGGER = logging.getLogger(__name__)


def setup_logger(verbose: bool = False) -> None:
    """Configure logger with appropriate level and format."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s", stream=sys.stdout)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Install Python packages using top-level requirements as constraints"
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("-f", "--force", action="store_true", help="Force reinstall all packages")
    return parser.parse_args()


def find_repo_root() -> Path:
    """Find the repository root from the current directory."""
    current = Path.cwd().resolve()
    while current != current.parent:
        if (current / "setup.py").exists() or (current / "pyproject.toml").exists():
            return current
        current = current.parent
    msg = "Could not find repository root (no setup.py or pyproject.toml found)"
    raise RuntimeError(msg)


def get_top_requirements(repo_root: Path) -> Path:
    """Get the top-level requirements file path."""
    top_requirements = repo_root / "requirements.txt"
    if not top_requirements.exists():
        msg = f"Top-level requirements.txt not found at {top_requirements}"
        raise FileNotFoundError(msg)
    return top_requirements


def install_local_package(
    package_path: Path, package_name: str | None = None, editable: bool = True, force: bool = False
) -> None:
    """Install a local package, optionally in editable mode with force reinstall."""
    try:
        if package_name:
            # First, try to uninstall any existing version
            subprocess.run(
                ["pip", "uninstall", "-y", package_name],
                check=False,
                capture_output=True,
                text=True,
            )

            # Clean build directories before installing
            build_dir = package_path / "build"
            if build_dir.exists():
                import shutil

                shutil.rmtree(build_dir)
                LOGGER.debug(f"Cleaned build directory: {build_dir}")

        install_args = ["pip", "install"]
        if force:
            install_args.append("--force-reinstall")
        if editable:
            install_args.extend(["-e", "."])
        else:
            install_args.append(".")

        subprocess.run(
            install_args,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(package_path),
        )
        LOGGER.debug(f"Successfully installed package from {package_path}")
    except subprocess.CalledProcessError as e:
        LOGGER.exception(f"Failed to install package from {package_path}: {e.stderr.strip()}")
        raise


def install_requirements(req_file: Path, constraint_file: Path, force: bool = False) -> None:
    """Install requirements using top-level requirements as constraints."""
    if not req_file.exists():
        msg = f"Local requirements.txt not found at {req_file}"
        raise FileNotFoundError(msg)

    LOGGER.info(f"Installing {req_file} with constraints from {constraint_file}")

    try:
        # Install requirements with constraints
        install_args = ["pip", "install", "-r", str(req_file), "-c", str(constraint_file)]
        if force:
            install_args.append("--force-reinstall")

        # If in debug mode, show output.
        output_setting = None if LOGGER.getEffectiveLevel() <= logging.DEBUG else subprocess.DEVNULL

        subprocess.run(
            install_args,
            check=True,
            stdout=output_setting,
            stderr=subprocess.PIPE,  # Keep capturing stderr for error handling
            text=True,
        )
        LOGGER.debug("Successfully installed requirements")

    except subprocess.CalledProcessError as e:
        LOGGER.exception(f"Failed to install requirements: {e.stderr.strip()}")
        raise


def main() -> None:
    """Install packages using top-level requirements as constraints."""
    args = parse_args()
    setup_logger(args.verbose)

    try:
        repo_root = find_repo_root()
        LOGGER.debug(f"Repository root: {repo_root}")

        top_requirements = get_top_requirements(repo_root)
        LOGGER.debug(f"Using constraints from: {top_requirements}")

        local_requirements = Path.cwd() / "requirements.txt"
        if not local_requirements.exists():
            msg = f"Local requirements.txt not found at {local_requirements}. Defaulting to "
            msg += "top-level requirements instead."
            LOGGER.warning(msg)
            local_requirements = top_requirements

        # Install repository package
        LOGGER.info("Step (1/2) - Repository package - installing...")
        install_local_package(repo_root, editable=True, force=args.force)

        # Install requirements
        LOGGER.info("Step (2/2) - Requirements       - installing...")
        install_requirements(local_requirements, top_requirements, force=args.force)

        LOGGER.info("Installation completed successfully")
    except Exception:
        LOGGER.exception("Error in install_requirements")
        sys.exit(1)


if __name__ == "__main__":
    main()
