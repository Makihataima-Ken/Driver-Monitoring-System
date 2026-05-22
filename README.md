# Driver Monitoring & Vehicle Safety System (DMS)

Real-time driver monitoring built for **Raspberry Pi** and PC development.  
Combines **MediaPipe FaceMesh**, **YOLOv8n**, and **OpenCV** for low-latency edge inference.

---

## Features

### Implemented (Demo v1)
| Event | Method | Status |
|---|---|---|
| `FATIGUE_DRIVING` | EAR via MediaPipe FaceMesh | ✅ Active |
| `DRIVER_YAWNS` | MAR via MediaPipe FaceMesh | ✅ Active |
| `DRIVER_UNDER_DISTRACTION` | Head pose (yaw/pitch) via PnP | ✅ Active |
| `NO_DRIVER` | Face absence counter | ✅ Active |
| `DRIVER_CALL` | YOLOv8n class 67 (cell phone) | ✅ Active |
| `DRIVER_SMOKE` | Custom weights (stub with base) | ⚠️ Needs fine-tuned weights |
| `SEAT_BELT_DETECTION` | Custom weights (stub with base) | ⚠️ Needs fine-tuned weights |
| `MOTION_DETECTION` | Frame differencing | ✅ Active (exterior) |

### Planned (v2)
`LANE_DEPARTURE`, `FRONT_CAR_COLLISION`, `PEDESTRIAN_COLLISION`, `DISTANCE_ALARM`, `COVER`, `REVERSE_CAM_*`

---

## Architecture

```
driver_monitor/
├── main.py                         # Entry point
├── demo_test.py                    # Headless smoke test
├── setup_weights.py                # Weight downloader
├── requirements.txt
├── src/
│   ├── camera/
│   │   └── camera_manager.py       # Threaded camera abstraction (OpenCV + Picamera2)
│   ├── config/
│   │   ├── settings.py             # Dataclass config system
│   │   └── default.yaml            # Tunable thresholds
│   ├── detectors/
│   │   ├── face_mesh_detector.py   # MediaPipe FaceMesh (EAR, MAR, head pose)
│   │   └── yolo_detector.py        # YOLOv8n wrapper
│   ├── pipelines/
│   │   ├── interior_pipeline.py    # Driver-facing camera pipeline
│   │   ├── exterior_pipeline.py    # Road-facing camera pipeline
│   │   └── system_pipeline.py      # Top-level orchestrator
│   ├── behaviors/
│   │   ├── driver_behaviors.py     # Fatigue, yawn, distraction, no-driver
│   │   └── yolo_behaviors.py       # Phone, smoke, seatbelt analyzers
│   ├── alerts/
│   │   ├── event_types.py          # DmsEvent + EventType enum
│   │   └── alert_manager.py        # Console + overlay + sound dispatch
│   └── utils/
│       ├── drawing.py              # OpenCV overlay helpers
│       ├── metrics.py              # FPS / latency / CPU / RAM monitor
│       ├── platform_detect.py      # Pi auto-detection
│       └── logger.py               # Centralized logging
└── weights/                        # Model weights (gitignored)
```

---

## Quick Start (PC / Development)

### 1. Clone & install dependencies

```bash
git clone <repo>
cd driver_monitor
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download weights

```bash
python setup_weights.py
```

### 3. Run the demo test (no camera needed)

```bash
python demo_test.py
```

### 4. Run with webcam

```bash
# Interior (driver-facing) pipeline
python main.py --pipeline interior

# Exterior (road-facing) pipeline
python main.py --pipeline exterior

# Both simultaneously
python main.py --pipeline both

# Headless / no display (for SSH / Pi without monitor)
python main.py --no-show

# Lower resolution for Pi
python main.py --width 320 --height 240 --fps-target 15
```

### Keyboard shortcuts (when display is open)
| Key | Action |
|-----|--------|
| `q` / `ESC` | Quit |
| `s` | Save screenshot |

---

## Raspberry Pi Setup

### Hardware Requirements
- Raspberry Pi 4 (2GB+ RAM recommended, 4GB for best experience)
- Pi Camera Module v2/v3 **or** USB webcam
- Heatsink + fan recommended for sustained inference

### OS Setup
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install system dependencies
sudo apt install -y python3-pip python3-venv libopencv-dev python3-opencv \
    libatlas-base-dev libhdf5-dev libjpeg-dev libpng-dev libwebp-dev \
    libtiff-dev libopenjp2-7-dev

# For Picamera2 (CSI camera)
sudo apt install -y python3-picamera2
```

### Python Environment on Pi
```bash
python3 -m venv venv --system-site-packages  # Include system opencv/picamera2
source venv/bin/activate
pip install -r requirements.txt
```

### Pi Camera (CSI)
```bash
# Enable camera in raspi-config
sudo raspi-config
# → Interface Options → Camera → Enable

# Test camera
libcamera-hello
```

### Pi-Optimised Config
Edit `src/config/default.yaml`:
```yaml
camera:
  width: 320      # Smaller resolution → faster
  height: 240
  fps_target: 15  # Target 15 FPS on Pi 4

mediapipe:
  refine_landmarks: false   # Save ~20% CPU
  max_faces: 1

yolo:
  imgsz: 320      # Critical for Pi performance
  device: cpu
```

### Run on Pi
```bash
# USB webcam
python3 main.py --width 320 --height 240 --fps-target 15

# Pi CSI camera (auto-detected)
python3 main.py --width 320 --height 240 --fps-target 15

# Force CSI camera
# Edit default.yaml: camera.use_picamera2: true

# Headless (SSH session, no monitor)
python3 main.py --no-show
```

### Pi Performance Notes
| Setting | Pi 4 (4GB) | Pi 4 (2GB) | Pi Zero 2W |
|---------|-----------|-----------|-----------|
| Resolution | 320×240 | 320×240 | 240×180 |
| FPS target | 15–20 | 10–15 | 5–8 |
| YOLO imgsz | 320 | 320 | 224 |
| MediaPipe refine | false | false | false |
| Expected real FPS | 12–18 | 8–12 | 4–6 |

**Tips:**
- Run `python main.py --no-show` to save CPU used by display rendering
- Use `--fps-target 15` or lower to avoid overheating
- Add a heatsink — sustained inference without cooling throttles the Pi
- YOLO runs every 3rd frame by default (`_yolo_frame_interval = 3`)

---

## Extending with Custom Weights

### Fine-tuned YOLO for phone / cigarette / seatbelt

1. Train YOLOv8n on your custom dataset (see Roboflow or custom labeling)
2. Export to `.pt` or `.onnx`
3. Update config:
```yaml
yolo:
  model_path: weights/custom_dms.pt
  classes_of_interest: [0, 1, 2, 3]  # Your custom class IDs
```
4. Update class maps in `src/detectors/yolo_detector.py`:
```python
CUSTOM_CLASSES = {
    0: "phone",
    1: "cigarette",
    2: "no_seatbelt",
    3: "seatbelt",
}
```

### Adding a New Event

1. Add the `EventType` to `src/alerts/event_types.py`
2. Create a new analyzer in `src/behaviors/`
3. Wire it into the appropriate pipeline in `src/pipelines/`
4. Done — the alert system handles dispatch automatically

---

## Troubleshooting

**`ImportError: mediapipe`**  
→ `pip install mediapipe`

**`ImportError: ultralytics`**  
→ `pip install ultralytics`

**Camera index 0 not found**  
→ Try `--camera 1` or `--camera 2`

**Very low FPS on Pi**  
→ Lower resolution: `--width 320 --height 240 --fps-target 10`  
→ Ensure heatsink is installed and throttling isn't active: `vcgencmd get_throttled`

**Picamera2 not detected**  
→ Enable camera in `raspi-config` and reboot  
→ Install: `sudo apt install python3-picamera2`

---

## License

MIT — see LICENSE file.
