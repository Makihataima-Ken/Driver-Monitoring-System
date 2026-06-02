"""
Detect whether we're running on a Raspberry Pi.
Used by CameraManager to choose the right backend.
"""

import platform
import re
import shutil
import subprocess


def is_raspberry_pi() -> bool:
    """Return True if the current platform is a Raspberry Pi."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            content = f.read()
        return "Raspberry Pi" in content or "BCM" in content
    except FileNotFoundError:
        pass

    machine = platform.machine().lower()
    return machine.startswith("arm") or machine.startswith("aarch")


def picamera2_available() -> bool:
    """Return True if picamera2 library is importable."""
    try:
        import picamera2  # noqa
        return True
    except ImportError:
        return False


def rpicam_camera_available() -> bool:
    """Return True if rpicam apps can see at least one CSI camera."""
    command = shutil.which("rpicam-hello")
    if command is None or shutil.which("rpicam-vid") is None:
        return False

    try:
        result = subprocess.run(
            [command, "--list-cameras"],
            capture_output=True,
            check=False,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return False

    output = f"{result.stdout}\n{result.stderr}"
    return re.search(r"^\s*\d+\s*:", output, flags=re.MULTILINE) is not None
