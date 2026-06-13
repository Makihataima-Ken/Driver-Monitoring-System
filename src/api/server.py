"""FastAPI server for exposing DMS alerts over HTTP."""

from __future__ import annotations

import threading
from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from src.alerts.alert_manager import AlertManager
from src.alerts.event_types import DmsEvent, EventType, Severity


class AlertRequest(BaseModel):
    event_type: EventType
    severity: Severity = Severity.WARNING
    message: str = ""
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)


def _event_to_dict(event: DmsEvent) -> dict[str, Any]:
    return {
        "event_type": event.event_type.value,
        "severity": event.severity.value,
        "message": event.message,
        "confidence": event.confidence,
        "timestamp": event.timestamp,
        "metadata": event.metadata,
    }


class ApiServer:
    def __init__(
        self,
        alert_manager: AlertManager,
        host: str = "0.0.0.0",
        port: int = 8000,
    ):
        self._alert_manager = alert_manager
        self._host = host
        self._port = port
        self._app = FastAPI(title="Driver Monitoring API")
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

        self._register_routes()

    def _register_routes(self) -> None:
        @self._app.get("/health")
        def health() -> dict[str, str]:
            return {"status": "ok"}

        @self._app.get("/alerts/recent")
        def recent_alerts() -> list[dict[str, Any]]:
            return [_event_to_dict(evt) for evt in self._alert_manager.get_recent_events()]

        @self._app.post("/alerts")
        def create_alert(request: AlertRequest) -> dict[str, Any]:
            event = DmsEvent(
                event_type=request.event_type,
                severity=request.severity,
                message=request.message or request.event_type.value,
                confidence=request.confidence,
                metadata=request.metadata,
            )
            self._alert_manager.dispatch(event)
            return _event_to_dict(event)

    def start(self) -> None:
        config = uvicorn.Config(
            app=self._app,
            host=self._host,
            port=self._port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._server is None:
            return

        self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
