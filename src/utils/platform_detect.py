"""
Detect whether we're running on a Raspberry Pi.
Used by CameraManager to choose the right backend.
"""

import platform
import os


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
