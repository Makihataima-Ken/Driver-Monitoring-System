# Decision Layer Port Plan

Status: planning only. Do not implement WP1-WP8 until the user replies
`approved`. Do not implement WP9 without a second, separate approval.

This plan is grounded in the current `Driver-Monitoring-System` repository. I
did not find a local `dms_final_system` reference checkout, so the architecture
below follows the requested B-system behavior contract while preserving the
current repo's Pi 4 performance advantages:

- Keep `src/detectors/yolo_detector.py` NCNN/exported-backend path, `imgsz:
  192`, `max_det=8`, warmup, and ROI offset remapping.
- Keep `src/pipelines/inference_runner.py` face ROI cropping, adaptive YOLO
  interval, no-face YOLO skip, and threaded capture/inference topology.
- Keep `src/detectors/face_mesh_detector.py` `process_width: 256` downscale
  and full-resolution landmark remapping.
- Keep `main.py` CLI flags: `--no-show`, `--width`, `--height`,
  `--fps-target`, `--yolo-interval`, `--no-yolo`, `--no-roi`,
  `--process-width`, `--profile`, and FastAPI startup flags working.
- No `torch` or `pandas` runtime dependency for the Pi path.

## WP1 - Contracts And Monotonic Timing

Purpose: create typed packets/results and remove frame-counted timing as a
decision dependency.

### New Files

- `src/contracts.py`

### Modified Files

- `src/camera/camera_manager.py`
- `src/pipelines/inference_runner.py`
- `src/pipelines/interior_result_processor.py`
- `src/pipelines/system_pipeline.py`
- `src/detectors/face_mesh_detector.py`
- `src/behaviors/driver_behaviors.py`
- `src/behaviors/yolo_behaviors.py`
- `src/alerts/event_types.py`
- `src/config/settings.py`
- `src/config/default.yaml`
- `src/utils/drawing.py`
- `src/api/server.py`

### New Config Keys And Defaults

Initial seconds-based compatibility keys, later refined by WP2-WP4:

```yaml
decision:
  version: decision.v1
  shadow_mode: false
  timing:
    max_clock_skew_warn_sec: 0.050
  fatigue:
    legacy_absolute_ear_threshold: 0.22
    closed_eye_min_sec: 1.00
  yawn:
    legacy_mar_threshold: 0.60
    mouth_open_min_sec: 0.80
  distraction:
    yaw_threshold_deg: 30.0
    pitch_threshold_deg: 20.0
    sustained_pose_sec: 1.30
  no_driver:
    absent_sec: 2.00
  violations:
    phone:
      legacy_hold_sec: 0.35
```

`mediapipe.ear_consec_frames`, `mediapipe.mar_consec_frames`, and
`mediapipe.distraction_consec_frames` become deprecated aliases for one release
only. If present, load them with a warning and convert using the configured
`camera.fps_target`; do not use them internally.

### Public Interfaces

Type signatures only:

```python
CONTRACT_VERSION: str

@dataclass(slots=True)
class FramePacket:
    frame_id: int
    utc_timestamp: float
    monotonic_sec: float
    frame: np.ndarray

DriverDistanceStatus = Literal["OK", "TOO_FAR", "NO_FACE"]

@dataclass(slots=True)
class FaceSignal:
    frame_id: int
    monotonic_sec: float
    face_present: bool
    ear: float | None
    left_ear: float | None
    right_ear: float | None
    mar: float | None
    yaw: float | None
    pitch: float | None
    roll: float | None
    face_bbox: tuple[int, int, int, int] | None
    landmarks_px: tuple[tuple[int, int], ...] | None
    frame_width: int
    frame_height: int
    relative_ear: float | None = None
    relative_left_ear: float | None = None
    relative_right_ear: float | None = None
    relative_mar: float | None = None
    face_width_ratio: float | None = None
    interocular_distance_px: float | None = None
    left_eye_width_px: float | None = None
    right_eye_width_px: float | None = None
    eye_resolution_valid: bool = False
    driver_distance_status: DriverDistanceStatus = "NO_FACE"
    binocular_consistent: bool | None = None
    quality: float = 0.0
    details: Mapping[str, object] = field(default_factory=dict)

@dataclass(slots=True)
class EvidenceEvent:
    kind: str
    monotonic_sec: float
    confidence: float
    ttl_sec: float
    source: str
    value: float | str | bool | None = None
    details: Mapping[str, object] = field(default_factory=dict)

    def is_fresh(self, now: float) -> bool: ...

@dataclass(slots=True)
class ObjectDetection:
    frame_id: int
    monotonic_sec: float
    class_id: int
    label: str
    confidence: float
    bbox: tuple[int, int, int, int]
    source: str
    details: Mapping[str, object] = field(default_factory=dict)

@dataclass(slots=True)
class FusionDecision:
    frame_id: int
    monotonic_sec: float
    driver_state: str
    alarm_level: str
    violations: tuple[str, ...]
    state_entry_reason: str
    current_reason_codes: tuple[str, ...]
    evidence_used: tuple[str, ...]
    trustworthy: bool
    details: Mapping[str, object] = field(default_factory=dict)

@dataclass(slots=True)
class AlarmStatus:
    monotonic_sec: float
    level: str
    active_episode_id: str | None
    pattern: str | None
    suppressed: bool
    next_reminder_sec: float | None
    details: Mapping[str, object] = field(default_factory=dict)

@dataclass(slots=True)
class IncidentRecord:
    incident_id: str
    start_monotonic_sec: float
    end_monotonic_sec: float | None
    highest_driver_state: str
    highest_alarm_level: str
    violations: tuple[str, ...]
    reason_codes: tuple[str, ...]
    contract_version: str
    decision_version: str
    event_version: str
    video_path: str | None
    video_sha256: str | None
    details: Mapping[str, object] = field(default_factory=dict)
```

Queue and processor signatures:

```python
class CameraManager:
    @property
    def frame_queue(self) -> queue.Queue[FramePacket]: ...
    def read(self) -> FramePacket | None: ...

class FaceMeshDetector:
    def process(self, packet: FramePacket) -> FaceSignal | None: ...

@dataclass(frozen=True, slots=True)
class InferenceResult:
    packet: FramePacket
    face_signal: FaceSignal | None
    yolo_detections: list[ObjectDetection]
    yolo_skipped: bool
    face_ms: float
    yolo_ms: float
    total_ms: float

class InteriorResultProcessor:
    def process(self, result: InferenceResult) -> tuple[list[DmsEvent], np.ndarray]: ...
```

### Deleted Or Replaced Existing Code

- Replace `queue.Queue[np.ndarray]` between `CameraManager` and
  `InferenceRunner` with `queue.Queue[FramePacket]`.
- Replace `InferenceResult.frame` with `InferenceResult.packet.frame`.
- Replace the 7-field `FaceResult` `NamedTuple` in
  `src/detectors/face_mesh_detector.py` with `FaceSignal`.
- Replace frame counters in `FatigueAnalyzer`, `YawnAnalyzer`,
  `DistractionAnalyzer`, `NoDriverAnalyzer`, and `PhoneCallAnalyzer` with
  monotonic-duration state.
- Keep `time.perf_counter()` for profiling and latency measurement, but forbid
  `time.time()` in decision code. Wall-clock `utc_timestamp` is captured once in
  the camera layer for logs/API compatibility; decisions use
  `FramePacket.monotonic_sec`.

### Risk Analysis

- Mixed-clock risk: event timestamps currently default to `time.time()`.
  Mitigation: route all decision timestamps through `FramePacket.monotonic_sec`;
  allow wall-clock only in UI, filenames, and API serialization.
- Compatibility risk: `InteriorPipeline` and `InteriorResultProcessor` duplicate
  behavior wiring. Mitigation: update both or retire `InteriorPipeline` only if
  nothing imports it; keep CLI behavior unchanged.
- Queue type risk: consumers may still assume raw arrays. Mitigation: add tests
  and update `bench_interior.py`/demo code together with runtime code in WP1.
- Safety risk: converting frames to seconds using `fps_target` can preserve old
  latency but not necessarily good behavior. Mitigation: mark converted values
  as migration defaults only; WP5 tune gate owns final thresholds.

## WP2 - Calibrator, Relative EAR, And Pixel Geometry Gates

Purpose: make fatigue detection driver-relative and reject geometrically unsafe
eye observations while preserving the FaceMesh downscale optimization.

### New Files

- `src/calibration/__init__.py`
- `src/calibration/personal_calibrator.py`

### Modified Files

- `src/contracts.py`
- `src/detectors/face_mesh_detector.py`
- `src/pipelines/interior_result_processor.py`
- `src/config/settings.py`
- `src/config/default.yaml`
- `src/utils/drawing.py`
- `README.md`

### New Config Keys And Defaults

```yaml
calibration:
  enabled: true
  target_sec: 10.0
  max_sec: 15.0
  minimum_good_frames: 75
  trim_low_quantile: 0.10
  trim_high_quantile: 0.90
  max_pose_deviation_deg: 12.0
  max_ear_cv: 0.18
  max_eye_asymmetry_ratio: 0.25
  min_quality: 0.70
  require_eye_resolution_valid: true
  persist_path: null

face_quality:
  min_face_width_ratio: 0.20
  min_interocular_distance_px: 50.0
  min_eye_width_px: 24.0
  binocular_consistency_max_asymmetry: 0.35

decision:
  fatigue:
    relative_ear_closed_threshold: 0.72
    relative_ear_partial_threshold: 0.85
  yawn:
    relative_mar_open_threshold: 1.60
```

`decision.fatigue.legacy_absolute_ear_threshold` remains telemetry-only after
calibration is ready.

### Public Interfaces

```python
CALIBRATION_VERSION: str

@dataclass(frozen=True, slots=True)
class CalibrationResult:
    version: str
    created_monotonic_sec: float
    sample_count: int
    ear_baseline: float
    left_ear_baseline: float
    right_ear_baseline: float
    mar_baseline: float
    pitch_baseline: float
    ear_cv: float
    eye_asymmetry_ratio: float
    accepted: bool
    rejection_reason: str | None
    details: Mapping[str, object] = field(default_factory=dict)

class PersonalCalibrator:
    def __init__(self, cfg: CalibrationConfig) -> None: ...
    @property
    def ready(self) -> bool: ...
    @property
    def result(self) -> CalibrationResult | None: ...
    def update(self, signal: FaceSignal) -> CalibrationResult | None: ...
    def apply(self, signal: FaceSignal) -> FaceSignal: ...
    def reset(self, monotonic_sec: float) -> None: ...
```

Face geometry helper signatures:

```python
def compute_face_quality(signal: FaceSignal, cfg: FaceQualityConfig) -> FaceSignal: ...
def compute_relative_signal(signal: FaceSignal, result: CalibrationResult) -> FaceSignal: ...
```

### Deleted Or Replaced Existing Code

- Replace fatigue decisions based on absolute `ear_threshold: 0.22` with
  `relative_ear`.
- Keep absolute `ear`, `left_ear`, and `right_ear` in `FaceSignal` for
  telemetry and diagnostics.
- Compute `face_width_ratio`, `interocular_distance_px`,
  `left_eye_width_px`, `right_eye_width_px`, `eye_resolution_valid`,
  `driver_distance_status`, and `binocular_consistent` from full-resolution
  `landmarks_px`.
- Do not hard-reject frames on `binocular_consistent`; expose it as a
  diagnostic because normal blinks can be asymmetric by one or two frames.

### Risk Analysis

- Pixel gates under 256px downscale: the detector currently infers on a
  256px-wide frame and maps normalized landmarks back to full-resolution
  pixels. Gates must use the mapped full-resolution coordinates, not the
  downscaled frame size. Mitigation: WP5 includes a test that creates a
  640x480 packet with `process_width=256` and proves a valid 60px interocular
  distance stays valid.
- Calibration false acceptance: sleepy startup, glasses glare, or occlusion can
  produce bad baselines. Mitigation: require good face quality, bounded head
  pose, trimmed quantiles, low EAR coefficient of variation, and low eye
  asymmetry.
- Calibration false rejection: some drivers have real asymmetry or camera
  placement bias. Mitigation: rejection leaves system in absolute-EAR fallback
  plus SHADOW telemetry until replay tuning confirms safer thresholds.
- Runtime cost: geometry is scalar distance math over already-computed
  landmarks. Estimated under 0.35 ms/frame.

## WP3 - Evidence Bus, PERCLOS, And Eye Observation State

Purpose: convert per-frame signals into durable physiological evidence with
TTLs, proper PERCLOS readiness, and dropout tolerance.

### New Files

- `src/events/__init__.py`
- `src/events/evidence_bus.py`
- `src/events/event_engine.py`

### Modified Files

- `src/contracts.py`
- `src/pipelines/interior_result_processor.py`
- `src/behaviors/driver_behaviors.py`
- `src/config/settings.py`
- `src/config/default.yaml`
- `src/utils/drawing.py`

### New Config Keys And Defaults

```yaml
events:
  version: events.v1
  max_bus_events: 256

eye_observation:
  max_eye_signal_gap_sec: 0.30
  lost_after_sec: 1.00
  require_eye_resolution_valid: true

event_engine:
  blink_min_sec: 0.08
  blink_max_sec: 0.40
  long_blink_min_sec: 0.50
  prolonged_eye_closure_sec: 1.50
  partial_closure_window_sec: 10.0
  partial_closure_min_total_sec: 3.0
  yawn_min_sec: 0.80
  yawn_max_sec: 6.00
  head_nod_pitch_delta_deg: 12.0
  head_nod_min_sec: 0.15
  rolling_counter_window_sec: 60.0

perclos:
  window_sec: 30.0
  min_valid_observation_sec: 20.0
  min_coverage_ratio: 0.67
  closed_relative_ear_threshold: 0.72
```

### Public Interfaces

```python
EVENT_VERSION: str

class EvidenceBus:
    def __init__(self, max_events: int = 256) -> None: ...
    def publish(self, events: Iterable[EvidenceEvent]) -> None: ...
    def drain(self, now: float | None = None) -> list[EvidenceEvent]: ...

EyeObservationStatus = Literal["VALID", "GRACE", "LOST"]
EyeState = Literal["OPEN", "CLOSED"]
MouthState = Literal["CLOSED", "OPEN"]

@dataclass(frozen=True, slots=True)
class PerclosStatus:
    perclos_30s: float | None
    valid_observation_sec_30s: float
    perclos_coverage_30s: float
    ready: bool

@dataclass(frozen=True, slots=True)
class EventEngineOutput:
    events: tuple[EvidenceEvent, ...]
    eye_state: EyeState
    mouth_state: MouthState
    eye_observation_status: EyeObservationStatus
    eye_evidence_trustworthy: bool
    current_closure_sec: float
    current_mouth_open_sec: float
    blink_count_60s: int
    yawn_count_60s: int
    perclos: PerclosStatus
    reason_codes: tuple[str, ...]

class EventEngine:
    def __init__(self, cfg: EventEngineConfig) -> None: ...
    def update(self, signal: FaceSignal | None, now: float) -> EventEngineOutput: ...
    def reset(self, now: float) -> None: ...
```

### Deleted Or Replaced Existing Code

- Replace analyzer hard resets on `face is None` with evidence TTL expiry and
  the eye observation `VALID -> GRACE -> LOST` machine.
- Replace `_consec = 0` on one good frame with duration-based open/closed
  state transitions.
- Add blink, long blink, prolonged closure, sustained partial closure, yawn,
  head nod, rolling 60s counters, and PERCLOS.

### Risk Analysis

- PERCLOS coverage gating: ungated PERCLOS can label missing observations as
  open or closed. Mitigation: report `perclos_30s` as not ready unless
  `valid_observation_sec_30s >= 20.0` and `perclos_coverage_30s >= 0.67`.
- GRACE timer freezing vs resetting: a brief dropout should not erase a
  candidate microsleep, but it also must not accumulate closure time while the
  eye is unobserved. Mitigation: freeze candidate timers in GRACE, resume if a
  trustworthy signal returns before `max_eye_signal_gap_sec`, and transition to
  LOST after the configured lost duration.
- TTL risk: evidence can linger too long and drive stale fusion. Mitigation:
  every `EvidenceEvent` has a TTL and `EvidenceBus.drain()` filters by
  `is_fresh(now)`.
- Ring-buffer boundary risk: PERCLOS window edge math is easy to get off by one
  sample. Mitigation: tests pin exact valid/invalid durations using synthetic
  monotonic timestamps.

## WP4 - Model-Free Fusion State Machine

Purpose: separate driver physiological state from behavior violations and
produce traceable, hysteretic decisions without requiring an ML model.

### New Files

- `src/fusion/__init__.py`
- `src/fusion/state_machine.py`

### Modified Files

- `src/contracts.py`
- `src/alerts/event_types.py`
- `src/pipelines/interior_result_processor.py`
- `src/pipelines/system_pipeline.py`
- `src/api/server.py`
- `src/config/settings.py`
- `src/config/default.yaml`
- `src/utils/drawing.py`

### New Config Keys And Defaults

```yaml
fusion:
  version: fusion.v1
  initial_state: UNKNOWN
  transition_cooldown_sec: 0.50
  grace_holds_candidate_timers: true
  post_critical_memory_sec: 60.0
  states:
    fatigue_warning:
      enter_persistence_sec: 1.00
      exit_persistence_sec: 3.00
      perclos_enter: 0.15
      perclos_exit: 0.10
      yawn_count_60s_enter: 2
      blink_count_60s_enter: 35
    drowsy:
      enter_persistence_sec: 1.00
      exit_persistence_sec: 5.00
      perclos_enter: 0.25
      perclos_exit: 0.18
      long_blink_count_60s_enter: 2
      closure_sec_enter: 1.00
    critical:
      bilateral_closure_sec_enter: 1.50
      recovery_open_eye_sec: 2.00
      latch_while_eye_lost: true
  optional_probability:
    enabled: false
    ewma_alpha: 0.20
    warning_enter: 0.55
    drowsy_enter: 0.70
    critical_enter: 0.85

violations:
  warning_level_only: true
  types:
    - PHONE_USE
    - SMOKING
    - EATING
    - DISTRACTION
    - NO_DRIVER
```

### Public Interfaces

```python
FUSION_VERSION: str

DriverState = Literal[
    "UNKNOWN", "NORMAL", "FATIGUE_WARNING", "DROWSY", "CRITICAL"
]
AlarmLevel = Literal["NONE", "INFO", "WARNING", "CRITICAL"]
ViolationType = Literal[
    "PHONE_USE", "SMOKING", "EATING", "DISTRACTION", "NO_DRIVER", "UNAVAILABLE"
]

STATE_RANK: Mapping[DriverState, int]

@dataclass(frozen=True, slots=True)
class FusionInput:
    frame_id: int
    monotonic_sec: float
    signal: FaceSignal | None
    event_output: EventEngineOutput
    evidence: tuple[EvidenceEvent, ...]
    probability: float | None = None

class FusionStateMachine:
    def __init__(self, cfg: FusionConfig) -> None: ...
    @property
    def state(self) -> DriverState: ...
    def update(self, inp: FusionInput) -> FusionDecision: ...
    def reset(self, now: float) -> None: ...
```

Legacy API bridge:

```python
def fusion_decision_to_legacy_events(decision: FusionDecision) -> list[DmsEvent]: ...
```

### Deleted Or Replaced Existing Code

- Replace flat `EventType.FATIGUE_DRIVING` and `Severity.CRITICAL` as the core
  driver-state model with `FusionDecision.driver_state` and
  `FusionDecision.alarm_level`.
- Keep legacy `DmsEvent` only as a compatibility transport for the existing API,
  overlay, and exterior pipeline during transition.
- Ensure `PHONE_USE`, `SMOKING`, `EATING`, and `DISTRACTION` remain violations.
  They may raise `alarm_level` to `WARNING`, but must not make
  `driver_state == DROWSY`.
- Make `CRITICAL` reachable from trusted sustained bilateral eye closure alone
  at about 1.5s, independent of any model.

### Risk Analysis

- CRITICAL latch/recovery correctness: the highest-risk state bug is unlatching
  during eye-signal loss or recovering on a single open frame. Mitigation:
  `CRITICAL` latches while eye observation is LOST and recovers only after
  continuously trustworthy OPEN eyes for `recovery_open_eye_sec`.
- Hysteresis risk: too little hysteresis chatters; too much delays recovery.
  Mitigation: separate enter/exit thresholds and longer downward persistence.
- GRACE semantics risk: resetting candidate timers in GRACE loses real
  microsleeps; accumulating them overstates danger. Mitigation: freeze timers,
  expose reason code `EYE_SIGNAL_GRACE_TIMER_FROZEN`, and test it.
- Traceability risk: unexplained transitions are unacceptable. Mitigation:
  every transition stores `state_entry_reason` and every frame reports
  `current_reason_codes`.
- Conflation risk: current `EventType`/`Severity` mixes phone use and drowsiness.
  Mitigation: driver states and violations are separate fields in
  `FusionDecision`.

## WP5 - Telemetry, Replay Harness, Tests, And Shadow Mode

Purpose: make the new stack measurable and tunable before audible alarms or
incident recording are trusted.

### New Files

- `src/telemetry/__init__.py`
- `src/telemetry/structured.py`
- `src/camera/replay_capture.py`
- `tools/replay_report.py`
- `tests/conftest.py`
- `tests/contracts/test_contracts.py`
- `tests/calibration/test_personal_calibrator.py`
- `tests/events/test_event_engine.py`
- `tests/events/test_evidence_bus.py`
- `tests/fusion/test_state_machine.py`
- `tests/violations/test_yolo_temporal_filter.py`
- `tests/pipelines/test_frame_packet_pipeline.py`

### Modified Files

- `src/pipelines/system_pipeline.py`
- `src/pipelines/interior_result_processor.py`
- `src/alerts/alert_manager.py`
- `src/config/settings.py`
- `src/config/default.yaml`
- `main.py`
- `README.md`
- `requirements.txt`
- `pyproject.toml`

### New Config Keys And Defaults

```yaml
telemetry:
  enabled: true
  mode: INFO
  log_dir: auto_tmpfs
  filename_prefix: dms_decision
  max_bytes: 5242880
  backup_count: 4
  live_status_rate_hz: 2.0
  include_landmarks_debug: false

replay:
  source_path: null
  report_dir: reports/replay
  write_debug_video: false
  score_window_sec: 30.0

deployment:
  shadow_mode: false
  suppress_audible_in_shadow: true
```

`auto_tmpfs` resolves to `/dev/shm/dms` on Raspberry Pi/Linux when writable and
falls back to `logs/dms`. Documentation must call out SD-card wear and recommend
tmpfs for DEBUG logs.

### Public Interfaces

```python
TelemetryMode = Literal["INFO", "DEBUG"]

class StructuredTelemetry:
    def __init__(self, cfg: TelemetryConfig) -> None: ...
    def emit(
        self,
        stage: str,
        event: str,
        payload: Mapping[str, object],
        *,
        monotonic_sec: float,
        frame_id: int | None = None,
        window_id: str | None = None,
        incident_id: str | None = None,
        level: str = "INFO",
    ) -> None: ...
    def live_status(self, status: Mapping[str, object], now: float) -> None: ...
    def close(self) -> None: ...

class ReplayCapture:
    def __init__(self, path: str, cfg: CameraConfig) -> None: ...
    def start(self) -> None: ...
    @property
    def frame_queue(self) -> queue.Queue[FramePacket]: ...
    def read(self) -> FramePacket | None: ...
    def stop(self) -> None: ...

def build_replay_report(args: argparse.Namespace) -> int: ...
```

### Deleted Or Replaced Existing Code

- No production deletion is required in WP5 beyond routing new decisions to
  telemetry and shadow suppression.
- Add pytest as a development dependency only.
- Add `ReplayCapture` with the same consumer-facing interface as
  `CameraManager`; do not fork the decision pipeline for replay.
- Shadow mode runs the new decision stack and logs all decisions while
  suppressing audible/local outputs. Console/log output remains explicit.

### Risk Analysis

- Telemetry I/O risk: DEBUG JSONL can wear SD cards and add latency.
  Mitigation: INFO default, size rotation, tmpfs default on Pi, rate-limited
  live status, and no per-frame landmark dumps unless DEBUG explicitly enables
  them.
- Replay fidelity risk: laptop replay can accidentally bypass timing semantics.
  Mitigation: `ReplayCapture` emits `FramePacket` and preserves monotonic
  deltas based on source FPS or recorded timestamps.
- Shadow safety risk: "shadow" must suppress audible/local alarms, not the
  decision logs. Mitigation: test alarm suppression and telemetry emission.
- Test scope risk: this is the stop-and-tune gate. Mitigation: do not proceed
  to WP6-WP8 until the tests below pass and real recordings are scored.

## WP6 - Alarm Controller

Purpose: replace cooldown-only alerting with stateful, acknowledged, explicit
alarm output modes.

### New Files

- `src/alerts/alarm_controller.py`
- `src/alerts/outputs.py`

### Modified Files

- `src/alerts/alert_manager.py`
- `src/alerts/event_types.py`
- `src/pipelines/system_pipeline.py`
- `src/api/server.py`
- `src/config/settings.py`
- `src/config/default.yaml`
- `README.md`

### New Config Keys And Defaults

```yaml
alarm:
  mode: LOG_ONLY
  master_gain: 0.70
  episode_merge_gap_sec: 2.00
  acknowledgement_open_eye_sec: 2.00
  repeat_escalation_window_sec: 300.0
  repeat_escalation_count: 3
  continuous_warning_escalation_sec: 20.0
  patterns:
    FATIGUE_WARNING:
      level: WARNING
      pattern: warning_double
      reminder_sec: 10.0
    DROWSY:
      level: WARNING
      pattern: urgent_triple
      reminder_sec: 5.0
    CRITICAL:
      level: CRITICAL
      pattern: critical_continuous
      reminder_sec: 2.0
    VIOLATION:
      level: WARNING
      pattern: warning_single
      reminder_sec: 12.0
```

`alarm.mode` replaces `alert.sound_enabled`. Compatibility loading maps
`sound_enabled: false` to `LOG_ONLY` and `sound_enabled: true` to `LOCAL`.

### Public Interfaces

```python
AlarmMode = Literal["LOG_ONLY", "LOCAL", "NULL"]

class AlarmOutput(Protocol):
    def play(self, status: AlarmStatus) -> None: ...
    def stop(self, episode_id: str | None = None) -> None: ...
    def close(self) -> None: ...

class NullAlarmOutput:
    def play(self, status: AlarmStatus) -> None: ...
    def stop(self, episode_id: str | None = None) -> None: ...
    def close(self) -> None: ...

class LogAlarmOutput:
    def play(self, status: AlarmStatus) -> None: ...
    def stop(self, episode_id: str | None = None) -> None: ...
    def close(self) -> None: ...

class LinuxAudioOutput:
    def __init__(self, cfg: AlarmConfig) -> None: ...
    def play(self, status: AlarmStatus) -> None: ...
    def stop(self, episode_id: str | None = None) -> None: ...
    def close(self) -> None: ...

class AlarmController:
    def __init__(self, cfg: AlarmConfig, output: AlarmOutput) -> None: ...
    def update(self, decision: FusionDecision, signal: FaceSignal | None) -> AlarmStatus: ...
    def acknowledge(self, now: float, reason: str) -> None: ...
    def get_status(self) -> AlarmStatus: ...
```

### Deleted Or Replaced Existing Code

- Replace per-event cooldown dispatch in `AlertManager` with
  `AlarmController.update(FusionDecision, FaceSignal | None)`.
- Keep listener/API support by exposing recent `AlarmStatus` and compatibility
  `DmsEvent` records.
- Replace implicit `sound_enabled: false` with explicit output modes.

### Risk Analysis

- Alarm fatigue risk: reminder cadence and repeat escalation can annoy drivers
  if too aggressive. Mitigation: tune cadence from WP5 replay statistics and
  start with `LOG_ONLY`.
- Acknowledgement risk: a driver should not clear an alarm by briefly opening
  eyes after a critical closure. Mitigation: sustained trustworthy open-eye
  acknowledgement uses the same signal quality gates as fusion recovery.
- Output risk: Linux audio dependencies vary across Pi installs. Mitigation:
  `LOG_ONLY` default, `NullAlarmOutput` for tests/shadow, and isolated
  `LinuxAudioOutput`.

## WP7 - YOLO Temporal Filter

Purpose: stabilize object-based behavior evidence in seconds and make class
coverage honest.

### New Files

- `src/violations/__init__.py`
- `src/violations/yolo_temporal_filter.py`

### Modified Files

- `src/behaviors/yolo_behaviors.py`
- `src/pipelines/interior_result_processor.py`
- `src/pipelines/inference_runner.py`
- `src/detectors/yolo_detector.py`
- `src/config/settings.py`
- `src/config/default.yaml`
- `src/telemetry/structured.py`
- `README.md`

### New Config Keys And Defaults

```yaml
yolo_temporal_filter:
  version: yolo_filter.v1
  refresh_ttl_sec: 1.00
  active_ttl_sec: 2.50
  classes:
    cell_phone:
      class_ids: [67]
      labels: [cell_phone, phone]
      confidence_threshold: 0.35
      window_sec: 1.50
      min_samples: 3
      activation_ratio: 0.60
      activation_persistence_sec: 0.50
      clear_persistence_sec: 2.00
      evidence_kind: PHONE_USE
    cigarette:
      labels: [cigarette, smoke, smoking]
      confidence_threshold: 0.45
      window_sec: 1.50
      min_samples: 3
      activation_ratio: 0.60
      activation_persistence_sec: 0.50
      clear_persistence_sec: 2.00
      evidence_kind: SMOKING
      required_weight: custom
    no_seatbelt:
      labels: [no_seatbelt, no_belt]
      confidence_threshold: 0.45
      window_sec: 2.00
      min_samples: 4
      activation_ratio: 0.60
      activation_persistence_sec: 1.00
      clear_persistence_sec: 3.00
      evidence_kind: SEATBELT_MISSING
      required_weight: custom
```

### Public Interfaces

```python
YOLO_FILTER_VERSION: str

@dataclass(frozen=True, slots=True)
class TemporalClassStatus:
    label: str
    active: bool
    unavailable: bool
    confidence: float
    sample_count: int
    activation_ratio: float
    last_seen_monotonic_sec: float | None
    reason_codes: tuple[str, ...]

class YoloTemporalFilter:
    def __init__(self, cfg: YoloTemporalFilterConfig) -> None: ...
    def update(
        self,
        detections: Sequence[ObjectDetection],
        now: float,
    ) -> tuple[tuple[EvidenceEvent, ...], tuple[TemporalClassStatus, ...]]: ...
    def reset(self, now: float) -> None: ...
```

### Deleted Or Replaced Existing Code

- Replace `PhoneCallAnalyzer._CONSEC_THRESHOLD = 5` with per-class temporal
  windows.
- Mark `SmokingAnalyzer` and `SeatbeltAnalyzer` as `UNAVAILABLE` at startup
  when the loaded base weights expose only COCO class 67 from
  `classes_of_interest: [67]`.
- Emit telemetry for unavailable classes instead of silently advertising dead
  features.

### Risk Analysis

- Adaptive interval risk: YOLO does not run every frame, and skipped frames
  currently reuse old detections. Mitigation: the filter samples only fresh
  YOLO outputs and uses seconds-based activation/clear windows.
- Class coverage risk: cigarette and seatbelt labels cannot fire with base
  COCO phone-only weights. Mitigation: explicit `UNAVAILABLE` status and
  startup telemetry.
- Small-object risk: cigarette detection likely needs larger `imgsz` than 192.
  Mitigation: do not degrade the existing 192px phone path; if custom smoking
  support is added later, prefer a second slower detector path.
- TTL risk: object evidence can stick after object disappears. Mitigation:
  asymmetric clear persistence plus periodic evidence refresh TTL.

## WP8 - Incident Recorder

Purpose: capture bounded pre/post evidence clips for warning/critical incidents
without blocking the decision thread.

### New Files

- `src/incidents/__init__.py`
- `src/incidents/recorder.py`

### Modified Files

- `src/contracts.py`
- `src/pipelines/system_pipeline.py`
- `src/alerts/alarm_controller.py`
- `src/telemetry/structured.py`
- `src/config/settings.py`
- `src/config/default.yaml`
- `README.md`

### New Config Keys And Defaults

```yaml
incident_recorder:
  enabled: true
  output_dir: incidents
  trigger_levels: [WARNING, CRITICAL]
  pre_sec: 10.0
  post_sec: 10.0
  merge_gap_sec: 5.0
  max_clip_sec: 60.0
  recorder_fps: 10
  width: 640
  height: 480
  max_queue_frames: 60
  encoder: auto_h264
  fallback_encoder: mp4v
  sha256_video: true
```

### Public Interfaces

```python
INCIDENT_RECORDER_VERSION: str

class IncidentRecorder:
    def __init__(self, cfg: IncidentRecorderConfig) -> None: ...
    def start(self) -> None: ...
    def observe_frame(self, packet: FramePacket) -> None: ...
    def observe_decision(self, decision: FusionDecision) -> IncidentRecord | None: ...
    def stop(self) -> None: ...
    def get_recent_records(self) -> list[IncidentRecord]: ...
```

### Deleted Or Replaced Existing Code

- No existing incident recorder exists. Add a bounded recorder behind the
  decision/alarm pipeline.
- Encoding and frame resizing happen off the decision thread.
- Recorder trigger is based on `FusionDecision.alarm_level` and violations, not
  raw detector events.

### Risk Analysis

- CPU/I/O risk: video encoding can destroy the Pi 4 performance envelope if it
  runs inline. Mitigation: nonblocking enqueue, bounded queue, worker thread,
  modest FPS/resolution, and hardware H.264 when available.
- Memory risk: 10s prebuffer at 640x480 can be over 130 MB at 15 FPS if stored
  as raw frames. Mitigation: configurable recorder FPS, bounded frame queue, and
  drop policy that preserves the newest pre-alert frames.
- Incident boundary risk: rapid repeated warnings can create many tiny clips.
  Mitigation: merge gaps, max clip length, and episode IDs from WP6.
- Privacy/storage risk: clips are sensitive. Mitigation: explicit output dir,
  bounded retention to be added before deployment policy, and SHA-256 recorded
  in `IncidentRecord`.

## WP9 - Optional ML Drowsiness Model

Purpose: optional probability input for fusion after the deterministic stack is
tuned and validated. This package requires separate approval even if WP1-WP8 are
approved.

### New Files

- `src/ml/__init__.py`
- `src/ml/feature_builder.py`
- `src/ml/lightgbm_predictor.py`
- `src/ml/drift_monitor.py`

### Modified Files

- `src/fusion/state_machine.py`
- `src/contracts.py`
- `src/telemetry/structured.py`
- `src/config/settings.py`
- `src/config/default.yaml`
- `tools/replay_report.py`
- `README.md`

### New Config Keys And Defaults

```yaml
ml_drowsiness:
  enabled: false
  bundle_path: null
  window_sec: 2.0
  stride_sec: 0.5
  target_samples: 30
  min_coverage: 0.80
  ewma_alpha: 0.20
  drift:
    enabled: true
    disable_on_ood: true
    max_missing_feature_ratio: 0.0
```

### Public Interfaces

```python
ML_DROWSINESS_VERSION: str

@dataclass(frozen=True, slots=True)
class FeatureWindow:
    window_id: str
    start_monotonic_sec: float
    end_monotonic_sec: float
    coverage: float
    names: tuple[str, ...]
    values: np.ndarray

class WindowedFeatureBuilder:
    def __init__(self, cfg: MlDrowsinessConfig) -> None: ...
    def update(self, output: EventEngineOutput, signal: FaceSignal | None) -> FeatureWindow | None: ...

class LightGbmDrowsinessPredictor:
    def __init__(self, bundle_path: str) -> None: ...
    def predict_proba(self, window: FeatureWindow) -> float | None: ...
    def validate_golden_sample(self) -> bool: ...

class DriftMonitor:
    def update(self, window: FeatureWindow) -> tuple[bool, tuple[str, ...]]: ...
```

### Deleted Or Replaced Existing Code

- No existing ML code is replaced.
- Do not enable `mediapipe.refine_landmarks`.
- Do not include `gaze_x`, `gaze_y`, or `gaze_zone` features.
- Do not add pandas or torch runtime dependencies.

### Risk Analysis

- Feature-distribution mismatch: this repo computes EAR from normalized
  landmarks without aspect correction, while a reference model may have been
  trained on pixel-space EAR. This is WP9's highest-risk failure mode.
  Mitigation: retrain without gaze features, validate feature schema against the
  booster count, run golden-sample parity at startup, and use drift/OOD to
  disable model contribution instead of silencing warnings.
- Model dominance risk: a bad probability could override reliable physiology.
  Mitigation: fusion accepts probability as optional evidence with bounded
  weight; deterministic CRITICAL closure remains independent.
- Runtime dependency risk: LightGBM wheels may be awkward on Pi. Mitigation:
  optional package, disabled by default, no pandas, and replay validation first.
- Approval risk: this work starts only after the second explicit approval.

## Dependency Graph And Shippability

Requested order:

```text
WP1 contracts/time
  -> WP2 calibration/geometry
    -> WP3 evidence/PERCLOS
      -> WP4 fusion
        -> WP5 telemetry/replay/tests/tune gate
          -> WP6 alarm controller
            -> WP7 YOLO temporal filter
              -> WP8 incident recorder
                -> WP9 optional ML
```

Technical dependencies:

```text
WP1 -> WP2 -> WP3 -> WP4
WP1 -> WP5
WP3 -> WP5
WP4 -> WP5
WP4 -> WP6
WP1 + WP3 -> WP7
WP1 + WP4 + WP6 -> WP8
WP2 + WP3 + WP4 + WP5 -> WP9
```

Independently shippable packages:

- WP1 is a mechanical foundation but should ship only with compatibility tests.
- WP2 can ship in SHADOW mode once relative EAR and pixel-gate telemetry are
  visible.
- WP3 can ship in SHADOW mode with event telemetry and no alarm changes.
- WP4 can ship in SHADOW/LOG_ONLY mode after deterministic state transitions
  pass tests.
- WP5 is a stop-and-tune gate, not just a feature package.
- WP6 can ship after WP5 when alarm thresholds are validated.
- WP7 is technically independent of WP6 once WP1/WP3 exist, but this plan keeps
  the requested order and ships it after the WP5 tune gate.
- WP8 ships only after WP6 episode IDs and alarm levels are stable.
- WP9 is optional and separately approved.

## Seconds-Based Config Schema

The full decision schema after WP1-WP8:

```yaml
decision:
  version: decision.v1
  shadow_mode: false
  timing:
    max_clock_skew_warn_sec: 0.050

calibration:
  enabled: true
  target_sec: 10.0
  max_sec: 15.0
  minimum_good_frames: 75
  trim_low_quantile: 0.10
  trim_high_quantile: 0.90
  max_pose_deviation_deg: 12.0
  max_ear_cv: 0.18
  max_eye_asymmetry_ratio: 0.25
  min_quality: 0.70
  require_eye_resolution_valid: true
  persist_path: null

face_quality:
  min_face_width_ratio: 0.20
  min_interocular_distance_px: 50.0
  min_eye_width_px: 24.0
  binocular_consistency_max_asymmetry: 0.35

eye_observation:
  max_eye_signal_gap_sec: 0.30
  lost_after_sec: 1.00
  require_eye_resolution_valid: true

event_engine:
  blink_min_sec: 0.08
  blink_max_sec: 0.40
  long_blink_min_sec: 0.50
  prolonged_eye_closure_sec: 1.50
  partial_closure_window_sec: 10.0
  partial_closure_min_total_sec: 3.0
  yawn_min_sec: 0.80
  yawn_max_sec: 6.00
  head_nod_pitch_delta_deg: 12.0
  head_nod_min_sec: 0.15
  rolling_counter_window_sec: 60.0

perclos:
  window_sec: 30.0
  min_valid_observation_sec: 20.0
  min_coverage_ratio: 0.67
  closed_relative_ear_threshold: 0.72

fusion:
  version: fusion.v1
  initial_state: UNKNOWN
  transition_cooldown_sec: 0.50
  grace_holds_candidate_timers: true
  post_critical_memory_sec: 60.0
  states:
    fatigue_warning:
      enter_persistence_sec: 1.00
      exit_persistence_sec: 3.00
      perclos_enter: 0.15
      perclos_exit: 0.10
      yawn_count_60s_enter: 2
      blink_count_60s_enter: 35
    drowsy:
      enter_persistence_sec: 1.00
      exit_persistence_sec: 5.00
      perclos_enter: 0.25
      perclos_exit: 0.18
      long_blink_count_60s_enter: 2
      closure_sec_enter: 1.00
    critical:
      bilateral_closure_sec_enter: 1.50
      recovery_open_eye_sec: 2.00
      latch_while_eye_lost: true
  optional_probability:
    enabled: false
    ewma_alpha: 0.20
    warning_enter: 0.55
    drowsy_enter: 0.70
    critical_enter: 0.85

violations:
  warning_level_only: true
  types: [PHONE_USE, SMOKING, EATING, DISTRACTION, NO_DRIVER]

yolo_temporal_filter:
  version: yolo_filter.v1
  refresh_ttl_sec: 1.00
  active_ttl_sec: 2.50
  classes:
    cell_phone:
      class_ids: [67]
      labels: [cell_phone, phone]
      confidence_threshold: 0.35
      window_sec: 1.50
      min_samples: 3
      activation_ratio: 0.60
      activation_persistence_sec: 0.50
      clear_persistence_sec: 2.00
      evidence_kind: PHONE_USE
    cigarette:
      labels: [cigarette, smoke, smoking]
      confidence_threshold: 0.45
      window_sec: 1.50
      min_samples: 3
      activation_ratio: 0.60
      activation_persistence_sec: 0.50
      clear_persistence_sec: 2.00
      evidence_kind: SMOKING
      required_weight: custom
    no_seatbelt:
      labels: [no_seatbelt, no_belt]
      confidence_threshold: 0.45
      window_sec: 2.00
      min_samples: 4
      activation_ratio: 0.60
      activation_persistence_sec: 1.00
      clear_persistence_sec: 3.00
      evidence_kind: SEATBELT_MISSING
      required_weight: custom

telemetry:
  enabled: true
  mode: INFO
  log_dir: auto_tmpfs
  filename_prefix: dms_decision
  max_bytes: 5242880
  backup_count: 4
  live_status_rate_hz: 2.0
  include_landmarks_debug: false

deployment:
  shadow_mode: false
  suppress_audible_in_shadow: true

alarm:
  mode: LOG_ONLY
  master_gain: 0.70
  episode_merge_gap_sec: 2.00
  acknowledgement_open_eye_sec: 2.00
  repeat_escalation_window_sec: 300.0
  repeat_escalation_count: 3
  continuous_warning_escalation_sec: 20.0

incident_recorder:
  enabled: true
  output_dir: incidents
  trigger_levels: [WARNING, CRITICAL]
  pre_sec: 10.0
  post_sec: 10.0
  merge_gap_sec: 5.0
  max_clip_sec: 60.0
  recorder_fps: 10
  width: 640
  height: 480
  max_queue_frames: 60
  encoder: auto_h264
  fallback_encoder: mp4v
  sha256_video: true
```

### Migration Table

| Current threshold | Current value | Assumed FPS | New key | New default | Notes |
|---|---:|---:|---|---:|---|
| `mediapipe.ear_consec_frames` in `default.yaml` | 15 frames | 15 FPS | `decision.fatigue.closed_eye_min_sec` | 1.00s | `settings.py` fallback is 20 frames at 20 FPS, also 1.00s. WP2 changes fatigue to `relative_ear`; absolute EAR remains telemetry. |
| `mediapipe.mar_consec_frames` in `default.yaml` | 12 frames | 15 FPS | `event_engine.yawn_min_sec` | 0.80s | `settings.py` fallback is 15 frames at 20 FPS, 0.75s. Use 0.80s as migration default. |
| `mediapipe.distraction_consec_frames` in `default.yaml` | 20 frames | 15 FPS | `decision.distraction.sustained_pose_sec` | 1.30s | `settings.py` fallback is 25 frames at 20 FPS, 1.25s. |
| `NoDriverAnalyzer._ABSENT_THRESHOLD` | 30 frames | 15 FPS | `decision.no_driver.absent_sec` | 2.00s | Old behavior stretched to 3.0-3.75s at 8-10 effective FPS. |
| `PhoneCallAnalyzer._CONSEC_THRESHOLD` | 5 frames | 15 FPS | `yolo_temporal_filter.classes.cell_phone.*` | 1.50s window, 0.50s activation | Current result processor reuses last YOLO detections on skipped frames, so this is not truly 5 fresh YOLO samples. WP7 replaces it with fresh-sample temporal filtering. |
| `SmokingAnalyzer._CONSEC_THRESHOLD` | 5 frames | 15 FPS | `yolo_temporal_filter.classes.cigarette.*` | unavailable unless custom weight | Base `classes_of_interest: [67]` cannot detect smoking. |
| `SeatbeltAnalyzer._CONSEC_THRESHOLD` | 8 frames | 15 FPS | `yolo_temporal_filter.classes.no_seatbelt.*` | unavailable unless custom weight | Base `classes_of_interest: [67]` cannot detect seatbelt/no-seatbelt. |

## WP5 Test Matrix

| Test name | Invariant |
|---|---|
| `tests/contracts/test_contracts.py::test_frame_packet_captures_single_monotonic_clock` | `FramePacket.monotonic_sec` is created at capture and survives camera -> inference -> processor. |
| `tests/contracts/test_contracts.py::test_evidence_event_ttl_expiry` | `EvidenceEvent.is_fresh(now)` returns false after `ttl_sec`. |
| `tests/pipelines/test_frame_packet_pipeline.py::test_camera_queue_emits_frame_packets` | `CameraManager.frame_queue` emits `FramePacket`, not raw `np.ndarray`. |
| `tests/pipelines/test_frame_packet_pipeline.py::test_inference_result_preserves_packet_id` | `InferenceResult.packet.frame_id` matches the input packet. |
| `tests/calibration/test_personal_calibrator.py::test_accepts_stable_good_calibration` | Stable good-quality face samples produce an accepted `CalibrationResult`. |
| `tests/calibration/test_personal_calibrator.py::test_rejects_high_ear_coefficient_of_variation` | Calibration rejects unstable EAR samples. |
| `tests/calibration/test_personal_calibrator.py::test_rejects_high_eye_asymmetry` | Calibration rejects excessive left/right baseline asymmetry. |
| `tests/calibration/test_personal_calibrator.py::test_rejects_pose_outliers` | Samples outside `max_pose_deviation_deg` do not enter the baseline. |
| `tests/calibration/test_personal_calibrator.py::test_apply_derives_relative_ear` | `relative_ear == ear / ear_baseline` and per-eye variants use their baselines. |
| `tests/calibration/test_personal_calibrator.py::test_pixel_gates_use_full_resolution_after_downscale` | Pixel gates use full-resolution landmarks after 256px FaceMesh downscale. |
| `tests/calibration/test_personal_calibrator.py::test_driver_distance_too_far_when_eye_pixels_small` | Small interocular/eye width sets `driver_distance_status == TOO_FAR`. |
| `tests/events/test_event_engine.py::test_blink_event_respects_min_max_duration` | Blink emits only inside configured min/max duration. |
| `tests/events/test_event_engine.py::test_long_blink_event_emits_after_threshold` | Long blink emits after `long_blink_min_sec`. |
| `tests/events/test_event_engine.py::test_prolonged_eye_closure_emits_after_threshold` | Sustained trusted closure emits `PROLONGED_EYE_CLOSURE`. |
| `tests/events/test_event_engine.py::test_yawn_event_respects_relative_mar_and_duration` | Yawn uses relative mouth ratio and min/max duration. |
| `tests/events/test_event_engine.py::test_head_nod_uses_pitch_delta_and_min_duration` | Head nod requires both pitch delta and duration. |
| `tests/events/test_event_engine.py::test_rolling_counts_are_60s_windowed` | Blink/yawn counters expire old events after 60s. |
| `tests/events/test_event_engine.py::test_perclos_not_ready_before_min_valid_observation` | PERCLOS is not ready before 20s valid observation. |
| `tests/events/test_event_engine.py::test_perclos_not_ready_below_coverage` | PERCLOS is not ready below 0.67 coverage. |
| `tests/events/test_event_engine.py::test_perclos_reports_ratio_when_ready` | Ready PERCLOS equals closed-valid duration divided by valid duration. |
| `tests/events/test_event_engine.py::test_eye_observation_valid_grace_lost_transitions` | Eye observation moves `VALID -> GRACE -> LOST` on configured gaps. |
| `tests/events/test_event_engine.py::test_grace_freezes_closure_timer` | GRACE freezes, rather than resets or increments, closure candidate timers. |
| `tests/events/test_evidence_bus.py::test_bus_drain_is_thread_safe_and_once_only` | Published evidence drains once without losing events under concurrent publish. |
| `tests/fusion/test_state_machine.py::test_unknown_to_normal_requires_trustworthy_signal` | Fusion does not enter NORMAL until eye evidence is trustworthy. |
| `tests/fusion/test_state_machine.py::test_warning_hysteresis_enter_exit` | FATIGUE_WARNING uses separate enter and exit persistence/thresholds. |
| `tests/fusion/test_state_machine.py::test_transition_cooldown_blocks_chatter` | Transitions cannot chatter inside cooldown. |
| `tests/fusion/test_state_machine.py::test_critical_reachable_from_sustained_closure_without_model` | 1.5s trusted bilateral closure reaches CRITICAL with no probability input. |
| `tests/fusion/test_state_machine.py::test_critical_latches_while_eye_lost` | CRITICAL does not recover during LOST eye observation. |
| `tests/fusion/test_state_machine.py::test_critical_recovery_requires_continuous_trustworthy_open_eyes` | Recovery requires sustained trusted OPEN eyes. |
| `tests/fusion/test_state_machine.py::test_post_critical_memory_recovers_to_fatigue_warning` | Post-critical recovery lands in fatigue memory before NORMAL. |
| `tests/fusion/test_state_machine.py::test_violations_raise_warning_without_drowsy_state` | PHONE_USE/SMOKING/EATING/DISTRACTION never make `driver_state == DROWSY`. |
| `tests/fusion/test_state_machine.py::test_reason_codes_present_on_every_decision` | Each decision includes `state_entry_reason` and `current_reason_codes`. |
| `tests/violations/test_yolo_temporal_filter.py::test_phone_activation_requires_window_samples_and_ratio` | Phone evidence requires min samples and activation ratio in the time window. |
| `tests/violations/test_yolo_temporal_filter.py::test_phone_clear_uses_asymmetric_persistence` | Active phone clears only after clear persistence. |
| `tests/violations/test_yolo_temporal_filter.py::test_filter_refreshes_evidence_before_ttl_expiry` | Active evidence refreshes before TTL expiry. |
| `tests/violations/test_yolo_temporal_filter.py::test_smoking_and_seatbelt_unavailable_with_coco_phone_only` | Unsupported classes emit UNAVAILABLE telemetry and do not silently fire. |

## Per-Frame CPU Budget

Estimates are for the decision layer only on Pi 4 and exclude existing
FaceMesh/YOLO inference, existing drawing, and optional WP9 model inference.

| Work package | Added per-frame cost estimate | Notes |
|---|---:|---|
| WP1 contracts/time | 0.05 ms | Dataclass packet access and monotonic timestamp propagation. |
| WP2 calibration/gates | 0.35 ms | Scalar landmark distances, relative ratios, quantile collection during startup. |
| WP3 event engine/PERCLOS | 0.35 ms | Ring buffers and duration state machines. |
| WP4 fusion | 0.10 ms | Small deterministic state machine. |
| WP5 telemetry/shadow | 0.20 ms | INFO mode amortized JSONL writes and 2 Hz live line; DEBUG can be higher and should use tmpfs. |
| WP6 alarm controller | 0.05 ms | Episode bookkeeping; audio playback is output-side and not per-frame compute. |
| WP7 YOLO temporal filter | 0.15 ms | Amortized over fresh YOLO samples, scalar windows only. |
| WP8 incident recorder | 0.10 ms | Nonblocking observe/enqueue only; encode/resize off decision thread. |
| Total WP1-WP8 | 1.35 ms | Leaves margin under the 2-4 ms budget. |
| WP9 optional ML | 0.50-1.50 ms | Separate approval; depends on LightGBM bundle and feature count. |

## Intentionally Not Ported

- Process-isolated YOLO. The current NCNN/exported-backend path avoids the
  PyTorch/EGL crash class and is the repo's performance advantage.
- Full-session evaluation video recording on the Pi. Full-session scoring is
  laptop-side via `tools/replay_report.py`; Pi incident recording stays bounded.
- MediaPipe iris refinement and gaze features. `refine_landmarks: false` stays
  because iris refinement costs about 20% CPU and WP9 must be retrained without
  gaze.
- Pandas and torch at runtime. Numpy-only feature building; torch remains only
  for export machines if needed by Ultralytics tooling.
- Degrading the existing phone detector path for cigarettes. If smoking support
  is added, use custom weights or a second slower detector path instead of
  raising `imgsz` on the 192px phone path.
- Exterior v2 safety features. This plan touches exterior code only where
  legacy event compatibility requires it.
- Audible alarms by default. `LOG_ONLY` is explicit until WP5 tuning supports
  switching to `LOCAL`.
- Fine-tuned multi-class behavior weights. The plan reports unavailable
  classes honestly but does not invent weights.

## Tune Gate After WP5

Do not proceed to WP6-WP8 until the following are measured on real recordings
with SHADOW mode enabled:

1. Performance envelope:
   - Sustained FPS on Pi 4 over at least 20 minutes: target remains 12-18 FPS.
   - Added decision-layer latency from telemetry: target <= 4 ms p95, <= 2 ms
     typical in INFO mode.
   - CPU temperature/throttling state and RSS memory on a 2GB Pi 4.

2. Calibration quality:
   - Acceptance rate across drivers/glasses/camera positions.
   - EAR baseline distribution, EAR coefficient of variation, and asymmetry
     rejection reasons.
   - Percentage of frames marked `TOO_FAR` or `eye_resolution_valid == false`.

3. Eye evidence:
   - Blink duration distribution.
   - Long-blink/prolonged-closure candidates against video review.
   - GRACE/LOST frequency and whether GRACE freezes preserve real closures.
   - PERCLOS readiness rate, `valid_observation_sec_30s`, coverage, and PERCLOS
     distribution for alert and non-alert clips.

4. Fusion:
   - Confusion review for `NORMAL`, `FATIGUE_WARNING`, `DROWSY`, and `CRITICAL`
     against labeled replay segments.
   - CRITICAL closure detection latency: target around 1.5s of trusted closure.
   - False CRITICAL cases during eye-signal LOST, glasses glare, and head turns.
   - Recovery behavior: no recovery until continuously trustworthy OPEN eyes.
   - Reason-code audit: every transition has an understandable reason.

5. Violations:
   - Phone temporal filter precision/recall and clear latency under adaptive
     YOLO intervals.
   - Startup telemetry confirms smoking/seatbelt unavailable with base COCO
     phone-only weights.

6. Alarm readiness:
   - Candidate warning/critical episode counts per hour.
   - Reminder cadence simulation from replay, before any audible output.
   - Driver acknowledgement simulation from sustained open-eye evidence.

7. Recorder readiness:
   - Estimated incident frequency and storage growth.
   - Pre/post clip boundaries from replay-generated incident timelines.
   - Confirm encoder choice and queue limits on the actual Pi.

Only after these measurements are reviewed should WP6-WP8 thresholds be
finalized for active deployment.
