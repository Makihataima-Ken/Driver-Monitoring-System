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
      --bg: #070f0d;
      --panel: #0e1f1b;
      --panel-soft: #12271f;
      --line: rgba(145, 220, 183, 0.14);
      --text: #ecf7ef;
      --muted: #8fb0a5;
      --accent: #b8f06a;
      --accent-soft: rgba(184, 240, 106, 0.14);
      --info: #6fd3ff;
      --warn: #ffb454;
      --danger: #ff6b5e;
      --ok: #7ee787;
      --radius: 14px;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--text);
      background:
        radial-gradient(900px circle at 12% -5%, rgba(56, 127, 96, 0.22), transparent 60%),
        radial-gradient(700px circle at 100% 0%, rgba(40, 90, 120, 0.16), transparent 55%),
        linear-gradient(160deg, #060d0b, #0c1916 60%, #060d0b);
      font-family: "Inter", "Segoe UI", "Trebuchet MS", Verdana, sans-serif;
      -webkit-font-smoothing: antialiased;
    }
    main { width: min(1240px, 95vw); margin: 0 auto; padding: 26px 0 40px; }

    header {
      display: flex; justify-content: space-between; align-items: center;
      gap: 18px; flex-wrap: wrap; margin-bottom: 20px;
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand .logo {
      width: 42px; height: 42px; border-radius: 12px; display: grid; place-items: center;
      background: var(--accent-soft); border: 1px solid var(--line); font-size: 22px;
    }
    h1 { margin: 0; font-size: clamp(1.25rem, 2.6vw, 1.8rem); letter-spacing: .04em; font-weight: 800; }
    .subtitle { margin: 2px 0 0; color: var(--muted); font-size: .82rem; }

    .status-wrap { display: flex; align-items: center; gap: 14px; flex-wrap: wrap; }
    .updated { color: var(--muted); font-size: .74rem; font-family: ui-monospace, monospace; }
    .status {
      display: inline-flex; align-items: center; gap: 9px;
      padding: 8px 14px; border: 1px solid var(--line); border-radius: 999px;
      background: var(--panel); color: var(--ok);
      font: 700 .76rem/1 ui-monospace, monospace; text-transform: uppercase; letter-spacing: .1em;
    }
    .status .dot {
      width: 9px; height: 9px; border-radius: 50%; background: currentColor;
      box-shadow: 0 0 0 0 currentColor; animation: pulse 1.6s infinite;
    }
    @keyframes pulse {
      0% { box-shadow: 0 0 0 0 rgba(126, 231, 135, .5); }
      70% { box-shadow: 0 0 0 8px rgba(126, 231, 135, 0); }
      100% { box-shadow: 0 0 0 0 rgba(126, 231, 135, 0); }
    }
    .status.running { color: var(--ok); }
    .status.waiting_for_camera { color: var(--warn); }
    .status.stopped, .status.disconnected { color: var(--danger); }
    .status.disconnected .dot { animation: none; }

    .grid { display: grid; grid-template-columns: minmax(0, 1fr) 340px; gap: 18px; align-items: start; }
    .panel {
      border: 1px solid var(--line); border-radius: var(--radius);
      background: linear-gradient(180deg, rgba(255, 255, 255, .015), transparent), var(--panel);
      box-shadow: 0 20px 50px rgba(0, 0, 0, .35);
    }
    .panel-head {
      display: flex; align-items: center; justify-content: space-between;
      padding: 12px 16px; border-bottom: 1px solid var(--line);
    }
    .panel-head h2 {
      margin: 0; font-size: .72rem; letter-spacing: .14em; text-transform: uppercase; color: var(--muted);
    }

    .video-shell {
      position: relative; display: grid; place-items: center;
      min-height: 360px; background: #020504; border-radius: var(--radius); overflow: hidden;
    }
    #stream { width: 100%; height: auto; display: block; }
    .video-badge {
      position: absolute; left: 14px; bottom: 12px; padding: 5px 10px; border-radius: 7px;
      color: var(--accent); background: rgba(0, 0, 0, .6); border: 1px solid var(--line);
      font: 700 .68rem ui-monospace, monospace; letter-spacing: .12em;
      display: inline-flex; align-items: center; gap: 7px;
    }
    .video-badge .dot { width: 7px; height: 7px; border-radius: 50%; background: var(--danger); animation: pulse 1.6s infinite; }

    .side { display: flex; flex-direction: column; gap: 18px; }

    .metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; padding: 14px; }
    .metric {
      padding: 12px; border: 1px solid var(--line); border-radius: 11px; background: var(--panel-soft);
      transition: border-color .2s;
    }
    .metric .k { color: var(--muted); font-size: .66rem; letter-spacing: .08em; text-transform: uppercase; }
    .metric .v { margin-top: 7px; color: var(--accent); font: 800 1.15rem ui-monospace, monospace; }
    .metric .v small { font-size: .62em; font-weight: 600; color: var(--muted); margin-left: 2px; }
    .metric.warn { border-color: rgba(255, 180, 84, .5); }
    .metric.warn .v { color: var(--warn); }
    .metric.danger { border-color: rgba(255, 107, 94, .5); }
    .metric.danger .v { color: var(--danger); }

    .events { padding: 14px; }
    #events { display: flex; flex-wrap: wrap; gap: 8px; }
    .chip {
      padding: 6px 11px; border-radius: 999px; font: 700 .72rem ui-monospace, monospace;
      border: 1px solid currentColor; background: rgba(255, 255, 255, .03);
    }
    .chip.ok { color: var(--ok); }
    .chip.warn { color: var(--warn); }
    .chip.danger { color: var(--danger); }
    .chip.empty { color: var(--muted); border-color: var(--line); }

    #logs {
      height: 280px; overflow: auto; margin: 0; padding: 8px 10px;
      display: flex; flex-direction: column; gap: 2px;
      font: .76rem/1.5 ui-monospace, "Cascadia Code", monospace;
    }
    .log-line {
      padding: 3px 9px; border-left: 3px solid var(--line); border-radius: 4px;
      white-space: pre-wrap; overflow-wrap: anywhere; color: #b8d4c8;
    }
    .log-line.info { border-color: rgba(111, 211, 255, .5); }
    .log-line.warn { border-color: var(--warn); color: #ffd9a6; background: rgba(255, 180, 84, .06); }
    .log-line.error { border-color: var(--danger); color: #ffc1ba; background: rgba(255, 107, 94, .08); }
    .log-line.debug { border-color: var(--line); color: var(--muted); }
    .log-empty { color: var(--muted); padding: 6px 9px; }

    @media (max-width: 880px) {
      .grid { grid-template-columns: 1fr; }
      .video-shell { min-height: 230px; }
      .metrics { grid-template-columns: 1fr 1fr 1fr; }
    }
    @media (max-width: 520px) {
      .metrics { grid-template-columns: 1fr 1fr; }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div class="brand">
        <div class="logo">&#128663;</div>
        <div>
          <h1>DRIVER MONITOR</h1>
          <p class="subtitle">Raspberry Pi &middot; live inference console</p>
        </div>
      </div>
      <div class="status-wrap">
        <span class="updated" id="updated">&mdash;</span>
        <span id="state" class="status disconnected"><span class="dot"></span><span id="state-text">connecting</span></span>
      </div>
    </header>

    <section class="grid">
      <article class="panel video-shell">
        <img id="stream" src="/video_feed" alt="Live driver monitoring stream">
        <span class="video-badge"><span class="dot"></span>LIVE &middot; MJPEG</span>
      </article>

      <div class="side">
        <section class="panel">
          <div class="panel-head"><h2>Telemetry</h2></div>
          <div class="metrics">
            <div class="metric" id="m-pfps"><div class="k">Processing FPS</div><div class="v" id="processing_fps">--</div></div>
            <div class="metric" id="m-cfps"><div class="k">Camera FPS</div><div class="v" id="camera_fps">--</div></div>
            <div class="metric"><div class="k">Target FPS</div><div class="v" id="target_fps">--</div></div>
            <div class="metric" id="m-lat"><div class="k">Latency</div><div class="v" id="latency_ms">--</div></div>
            <div class="metric" id="m-cpu"><div class="k">CPU</div><div class="v" id="cpu_percent">--</div></div>
            <div class="metric" id="m-ram"><div class="k">RAM</div><div class="v" id="ram_mb">--</div></div>
            <div class="metric"><div class="k">Resolution</div><div class="v" id="resolution">--</div></div>
            <div class="metric"><div class="k">Captured</div><div class="v" id="captured_frames">--</div></div>
            <div class="metric" id="m-drop"><div class="k">Dropped</div><div class="v" id="dropped_frames">--</div></div>
          </div>
        </section>

        <section class="panel events">
          <h2 style="margin:0 0 10px;font-size:.72rem;letter-spacing:.14em;text-transform:uppercase;color:var(--muted)">Active events</h2>
          <div id="events"><span class="chip empty">No active events</span></div>
        </section>
      </div>
    </section>

    <section class="panel" style="margin-top:18px">
      <div class="panel-head"><h2>Live logs</h2><span class="updated" id="log_file">logs/dms.log</span></div>
      <div id="logs"><div class="log-empty">Waiting for application logs&hellip;</div></div>
    </section>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    let lastUpdate = 0;

    const fmt = (n, d = 1) => (typeof n === "number" ? n.toFixed(d) : "--");

    function setMetricClass(elId, level) {
      const el = $(elId);
      if (!el) return;
      el.classList.remove("warn", "danger");
      if (level) el.classList.add(level);
    }

    function eventClass(label) {
      const l = String(label).toLowerCase();
      if (/drows|sleep|micro|eyes?_?closed|critical|alarm/.test(l)) return "danger";
      if (/yawn|distract|phone|head|gaze|look|no_?face|absent/.test(l)) return "warn";
      return "ok";
    }

    function logClass(line) {
      if (line.includes("[ERROR]") || line.includes("[CRITICAL]")) return "error";
      if (line.includes("[WARNING]")) return "warn";
      if (line.includes("[DEBUG]")) return "debug";
      return "info";
    }

    async function refresh() {
      try {
        const res = await fetch("/api/status", { cache: "no-store" });
        const d = await res.json();
        lastUpdate = Date.now();

        const st = $("state");
        st.className = "status " + d.state;
        $("state-text").textContent = d.state.replaceAll("_", " ");

        $("processing_fps").textContent = fmt(d.processing_fps);
        $("camera_fps").textContent = fmt(d.camera_fps);
        $("target_fps").textContent = fmt(d.target_fps);
        $("latency_ms").innerHTML = fmt(d.latency_ms, 0) + "<small>ms</small>";
        $("cpu_percent").innerHTML = fmt(d.cpu_percent, 0) + "<small>%</small>";
        $("ram_mb").innerHTML = fmt(d.ram_mb, 0) + "<small>MB</small>";
        $("resolution").textContent = d.resolution;
        $("captured_frames").textContent = d.captured_frames;
        $("dropped_frames").textContent = d.dropped_frames;

        setMetricClass("m-lat", d.latency_ms > 300 ? "danger" : d.latency_ms > 150 ? "warn" : null);
        setMetricClass("m-cpu", d.cpu_percent > 95 ? "danger" : d.cpu_percent > 85 ? "warn" : null);
        setMetricClass("m-drop", d.dropped_frames > 0 ? "warn" : null);
        setMetricClass("m-pfps", d.processing_fps && d.target_fps && d.processing_fps < d.target_fps * 0.5 ? "warn" : null);

        const ev = $("events");
        if (d.events && d.events.length) {
          ev.innerHTML = "";
          d.events.forEach((e) => {
            const c = document.createElement("span");
            c.className = "chip " + eventClass(e);
            c.textContent = String(e).replaceAll("_", " ");
            ev.appendChild(c);
          });
        } else {
          ev.innerHTML = '<span class="chip empty">No active events</span>';
        }
      } catch (e) {
        $("state").className = "status disconnected";
        $("state-text").textContent = "disconnected";
      }
    }

    async function refreshLogs() {
      try {
        const res = await fetch("/api/logs?limit=150", { cache: "no-store" });
        const d = await res.json();
        $("log_file").textContent = d.log_file;
        const box = $("logs");
        const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 60;
        if (!d.logs || !d.logs.length) {
          box.innerHTML = '<div class="log-empty">No logs yet.</div>';
          return;
        }
        box.innerHTML = "";
        d.logs.forEach((line) => {
          const row = document.createElement("div");
          row.className = "log-line " + logClass(line);
          row.textContent = line;
          box.appendChild(row);
        });
        if (atBottom) box.scrollTop = box.scrollHeight;
      } catch (e) {
        $("logs").innerHTML = '<div class="log-empty">Unable to load logs.</div>';
      }
    }

    function tickUpdated() {
      const el = $("updated");
      if (!lastUpdate) { el.textContent = "—"; return; }
      const s = Math.round((Date.now() - lastUpdate) / 1000);
      el.textContent = "updated " + (s <= 0 ? "just now" : s + "s ago");
    }

    refresh();
    refreshLogs();
    setInterval(refresh, 1000);
    setInterval(refreshLogs, 1500);
    setInterval(tickUpdated, 1000);
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
