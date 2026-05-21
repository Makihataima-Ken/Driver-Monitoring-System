"""
Alert and event data models.
All detectors produce DmsEvent objects consumed by the alert system.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import time


class EventType(str, Enum):
    # Interior
    FATIGUE_DRIVING = "FATIGUE_DRIVING"
    DRIVER_YAWNS = "DRIVER_YAWNS"
    DRIVER_CALL = "DRIVER_CALL"
    DRIVER_SMOKE = "DRIVER_SMOKE"
    DRIVER_UNDER_DISTRACTION = "DRIVER_UNDER_DISTRACTION"
    NO_DRIVER = "NO_DRIVER"
    SEAT_BELT_DETECTION = "SEAT_BELT_DETECTION"

    # Exterior (placeholders for v2)
    LANE_DEPARTURE = "LANE_DEPARTURE"
    FRONT_CAR_COLLISION = "FRONT_CAR_COLLISION"
    TOO_CLOSE_DISTANCE = "TOO_CLOSE_DISTANCE"
    PEDESTRIAN_COLLISION = "PEDESTRIAN_COLLISION"
    DISTANCE_ALARM = "DISTANCE_ALARM"

    # General
    MOTION_DETECTION = "MOTION_DETECTION"
    COVER = "COVER"
    REVERSE_CAM_COLLISION = "REVERSE_CAM_COLLISION"
    REVERSE_CAM_ZERO_SPEED = "REVERSE_CAM_ZERO_SPEED"


class Severity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"


@dataclass
class DmsEvent:
    event_type: EventType
    severity: Severity
    message: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)

    @classmethod
    def make(
        cls,
        event_type: EventType,
        severity: Severity = Severity.WARNING,
        message: str = "",
        confidence: float = 1.0,
        **metadata,
    ) -> "DmsEvent":
        return cls(
            event_type=event_type,
            severity=severity,
            message=message or event_type.value,
            confidence=confidence,
            metadata=metadata,
        )
