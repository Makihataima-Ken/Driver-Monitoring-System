"""Flask MJPEG dashboard for headless Raspberry Pi deployments."""

from __future__ import annotations

import logging
import threading
from typing import Protocol

import cv2
from flask import Flask, Response, jsonify, render_template_string, request
from werkzeug.serving import make_server

from src.config.settings import WebConfig
from src.utils.logger import get_log_path, get_recent_logs

logger = logging.getLogger("dms.web")


class StreamProvider(Protocol):
    @property
    def is_running(self) -> bool: ...

    def get_status(self) -> dict: ...

    def wait_for_stream_frame(
        self,
        last_sequence: int,
        timeout: float = 2.0,
    ) -> tuple[int, object | None]: ...


_DASHBOARD_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>DMS Live Monitor</title>
  <style>
    :root {
      --bg: #08110f;
      --panel: rgba(14, 31, 27, 0.88);
      --line: rgba(145, 220, 183, 0.2);
      --text: #ecf7ef;
      --muted: #99b9ae;
      --accent: #b8f06a;
      --warn: #ffb454;
      --danger: #ff7061;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(circle at 15% 15%, rgba(56, 127, 96, 0.28), transparent 35rem),
        linear-gradient(145deg, #07100e, #10211c 65%, #07100e);
      font-family: "Trebuchet MS", Verdana, sans-serif;
    }
    main { width: min(1180px, 94vw); margin: 0 auto; padding: 28px 0 38px; }
    header {
      display: flex; justify-content: space-between; align-items: end;
      gap: 18px; padding-bottom: 18px;
    }
    h1 { margin: 0; letter-spacing: .08em; font-size: clamp(1.4rem, 3vw, 2.3rem); }
    .subtitle { margin: 7px 0 0; color: var(--muted); font-size: .9rem; }
    .status {
      padding: 7px 12px; border: 1px solid var(--line); border-radius: 999px;
      color: var(--accent); background: var(--panel); font: 700 .78rem monospace;
      text-transform: uppercase; letter-spacing: .08em;
    }
    .status.waiting_for_camera { color: var(--warn); }
    .status.stopped { color: var(--danger); }
    .grid { display: grid; grid-template-columns: minmax(0, 1fr) 300px; gap: 16px; }
    .panel {
      border: 1px solid var(--line); border-radius: 14px; overflow: hidden;
      background: var(--panel); box-shadow: 0 18px 55px rgba(0, 0, 0, .24);
    }
    .video-shell {
      display: grid; min-height: 340px; place-items: center; background: #020504;
      position: relative;
    }
    #stream { width: 100%; height: auto; display: block; image-rendering: auto; }
    .video-label {
      position: absolute; left: 12px; bottom: 10px; padding: 5px 8px;
      color: var(--accent); background: rgba(0, 0, 0, .66); font: 700 .72rem monospace;
      letter-spacing: .08em;
    }
    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 1px; background: var(--line); }
    .metric { padding: 15px 13px; background: #10231e; min-height: 78px; }
    .metric span { display: block; color: var(--muted); font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; }
    .metric strong { display: block; margin-top: 7px; color: var(--accent); font: 700 1.18rem monospace; }
    .events { padding: 14px; border-top: 1px solid var(--line); min-height: 78px; }
    .events h2 { margin: 0 0 8px; color: var(--muted); font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }
    #events { color: var(--text); font: .82rem monospace; line-height: 1.55; }
    .log-panel { margin-top: 16px; padding: 14px; }
    .log-panel h2 { margin: 0 0 10px; color: var(--muted); font-size: .72rem; letter-spacing: .1em; text-transform: uppercase; }
    #logs {
      height: 220px; margin: 0; overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere;
      color: #bad9cc; background: rgba(0, 0, 0, .32); border: 1px solid var(--line);
      border-radius: 8px; padding: 10px; font: .76rem/1.45 monospace;
    }
    @media (max-width: 820px) {
      .grid { grid-template-columns: 1fr; }
      .video-shell { min-height: 220px; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <h1>DRIVER MONITOR</h1>
        <p class="subtitle">Raspberry Pi live inference console</p>
      </div>
      <div id="state" class="status">CONNECTING</div>
    </header>
    <section class="grid">
      <article class="panel video-shell">
        <img id="stream" src="/video_feed" alt="Live driver monitoring stream">
        <div class="video-label">LIVE / MJPEG</div>
      </article>
      <aside class="panel">
        <div class="metrics">
          <div class="metric"><span>Resolution</span><strong id="resolution">--</strong></div>
          <div class="metric"><span>Processing FPS</span><strong id="processing_fps">--</strong></div>
          <div class="metric"><span>Camera FPS</span><strong id="camera_fps">--</strong></div>
          <div class="metric"><span>Target FPS</span><strong id="target_fps">--</strong></div>
          <div class="metric"><span>Latency</span><strong id="latency_ms">--</strong></div>
          <div class="metric"><span>Captured</span><strong id="captured_frames">--</strong></div>
          <div class="metric"><span>Dropped</span><strong id="dropped_frames">--</strong></div>
          <div class="metric"><span>RAM</span><strong id="ram_mb">--</strong></div>
        </div>
        <div class="events">
          <h2>Active events</h2>
          <div id="events">None</div>
        </div>
      </aside>
    </section>
    <section class="panel log-panel">
      <h2>Live logs / <span id="log_file">logs/dms.log</span></h2>
      <pre id="logs">Waiting for application logs...</pre>
    </section>
  </main>
  <script>
    const setText = (id, value) => document.getElementById(id).textContent = value;
    async function refresh() {
      try {
        const response = await fetch("/api/status", { cache: "no-store" });
        const data = await response.json();
        const state = document.getElementById("state");
        state.textContent = data.state.replaceAll("_", " ");
        state.className = "status " + data.state;
        setText("resolution", data.resolution);
        setText("processing_fps", data.processing_fps.toFixed(1));
        setText("camera_fps", data.camera_fps.toFixed(1));
        setText("target_fps", data.target_fps.toFixed(1));
        setText("latency_ms", data.latency_ms.toFixed(1) + " ms");
        setText("captured_frames", data.captured_frames);
        setText("dropped_frames", data.dropped_frames);
        setText("ram_mb", data.ram_mb.toFixed(1) + " MB");
        setText("events", data.events.length ? data.events.join(" / ") : "None");
      } catch (error) {
        const state = document.getElementById("state");
        state.textContent = "DISCONNECTED";
        state.className = "status stopped";
      }
    }
    async function refreshLogs() {
      try {
        const response = await fetch("/api/logs?limit=120", { cache: "no-store" });
        const data = await response.json();
        const logs = document.getElementById("logs");
        const nearBottom = logs.scrollHeight - logs.scrollTop - logs.clientHeight < 48;
        setText("log_file", data.log_file);
        logs.textContent = data.logs.length ? data.logs.join("\\n") : "No logs yet.";
        if (nearBottom) logs.scrollTop = logs.scrollHeight;
      } catch (error) {
        document.getElementById("logs").textContent = "Unable to load logs.";
      }
    }
    refresh();
    refreshLogs();
    setInterval(refresh, 1000);
    setInterval(refreshLogs, 1000);
  </script>
</body>
</html>
"""


class WebServer:
    def __init__(self, cfg: WebConfig, provider: StreamProvider):
        self._cfg = cfg
        self._provider = provider
        self._server = None
        self._thread: threading.Thread | None = None
        self._app = Flask(__name__)
        self._register_routes()

    def _register_routes(self):
        @self._app.get("/")
        def dashboard():
            return render_template_string(_DASHBOARD_HTML)

        @self._app.get("/api/status")
        def status():
            return jsonify(self._provider.get_status())

        @self._app.get("/api/logs")
        def logs():
            try:
                limit = int(request.args.get("limit", 120))
            except ValueError:
                limit = 120
            return jsonify({"log_file": get_log_path(), "logs": get_recent_logs(limit)})

        @self._app.get("/video_feed")
        def video_feed():
            return Response(
                self._mjpeg_frames(),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

    def _mjpeg_frames(self):
        sequence = -1
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, self._cfg.jpeg_quality]
        while self._provider.is_running:
            sequence, frame = self._provider.wait_for_stream_frame(sequence)
            if frame is None:
                continue
            ok, jpeg = cv2.imencode(".jpg", frame, encode_params)
            if not ok:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + jpeg.tobytes()
                + b"\r\n"
            )

    def start(self):
        self._server = make_server(
            self._cfg.host,
            self._cfg.port,
            self._app,
            threaded=True,
        )
        self._thread = threading.Thread(
            target=self._server.serve_forever,
            daemon=True,
            name="dms-web-server",
        )
        self._thread.start()
        logger.info("Web dashboard listening on http://%s:%d", self._cfg.host, self._cfg.port)

    def stop(self):
        if self._server is None:
            return
        self._server.shutdown()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._server = None
        self._thread = None
        logger.info("Web dashboard stopped.")
