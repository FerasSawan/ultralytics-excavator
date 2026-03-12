import os
os.environ["QT_LOGGING_RULES"] = "qt.qpa.*=false"
os.environ["QT_QPA_FONTDIR"] = "/usr/share/fonts"
import cv2
import sys
import tty
import termios
import threading
import subprocess
import signal
import time
import json
import logging
from datetime import datetime
from ultralytics import YOLO
from flask import Flask, Response, jsonify

MODEL = "yolo26n.pt"
CSI_CAMERA = 0
CAM_WIDTH = 1280
CAM_HEIGHT = 720
CAM_FPS = 30
CONF = 0.25
OUTPUT_DIR = "video_outputs"
DISPLAY_WIDTH = 960
WEB_PORT = 5000

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL)

print(f"Starting CSI camera {CSI_CAMERA} via rpicam-vid (MJPEG pipe)...")
rpicam_proc = subprocess.Popen([
    "rpicam-vid", "-t", "0",
    "--camera", str(CSI_CAMERA),
    "--width", str(CAM_WIDTH), "--height", str(CAM_HEIGHT),
    "--framerate", str(CAM_FPS),
    "--codec", "mjpeg", "--quality", "80",
    "-o", "-",
    "--nopreview",
], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)

def read_mjpeg_frame():
    """Read one JPEG frame from the rpicam-vid MJPEG pipe."""
    buf = b""
    while True:
        chunk = rpicam_proc.stdout.read(4096)
        if not chunk:
            return None
        buf += chunk
        start = buf.find(b"\xff\xd8")
        if start == -1:
            buf = buf[-1:]
            continue
        end = buf.find(b"\xff\xd9", start + 2)
        if end != -1:
            jpg = buf[start:end + 2]
            buf = buf[end + 2:]
            import numpy as np
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                return frame

test_frame = read_mjpeg_frame()
if test_frame is None:
    print("ERROR: Cannot read from rpicam-vid")
    rpicam_proc.terminate()
    exit(1)
actual_w = test_frame.shape[1]
actual_h = test_frame.shape[0]
print(f"Camera opened: CSI cam {CSI_CAMERA} ({actual_w}x{actual_h} @ {CAM_FPS} FPS)")

lock = threading.Lock()
latest_frame = None
latest_detections = None
recording = False
writer = None
rec_frame_count = 0
rec_start_time = 0.0
rec_temp_path = ""
rec_final_path = ""
pending_action = None
running = True
human = False

fd = sys.stdin.fileno()
old_settings = termios.tcgetattr(fd)


def key_listener():
    global pending_action
    try:
        tty.setraw(fd)
        while running:
            ch = sys.stdin.read(1).lower()
            if ch == "r":
                pending_action = "start"
            elif ch == "s":
                pending_action = "stop"
            elif ch == "q":
                pending_action = "quit"
                break
    except Exception:
        pass


def draw_detections(frame, detections):
    """Draw cached YOLO bounding boxes onto a frame."""
    if detections is None:
        return frame
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = map(int, det["box"])
        label = f"{det['name']} {det['conf']:.2f}"
        color = det.get("color", (0, 255, 0))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw, y1), color, -1)
        cv2.putText(annotated, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return annotated


def capture_thread():
    """Read frames from camera at full speed and write to recording."""
    global latest_frame, recording, writer, rec_frame_count
    while running:
        frame = read_mjpeg_frame()
        if frame is None:
            break
        with lock:
            latest_frame = frame
            if recording and writer is not None:
                annotated = draw_detections(frame, latest_detections)
                writer.write(annotated)
                rec_frame_count += 1


def finalize_video(temp_path, final_path, frame_count, elapsed):
    """Re-mux the temp video at the true measured FPS so playback = real time."""
    if frame_count == 0 or elapsed == 0:
        print("No frames recorded.")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return
    actual_fps = frame_count / elapsed
    print(f"Finalizing: {frame_count} frames in {elapsed:.1f}s = {actual_fps:.2f} FPS")

    temp_cap = cv2.VideoCapture(temp_path)
    w = int(temp_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(temp_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(final_path, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (w, h))

    while True:
        ok, frame = temp_cap.read()
        if not ok:
            break
        out.write(frame)

    temp_cap.release()
    out.release()
    os.remove(temp_path)
    print(f"Saved: {final_path}")


web_display_frame = None

app = Flask(__name__)
logging.getLogger("werkzeug").setLevel(logging.ERROR)

@app.route("/")
def index():
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>YOLO Live</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0f0f0f; color: #fff; font-family: 'Segoe UI', system-ui, sans-serif;
         display: flex; flex-direction: column; align-items: center; min-height: 100vh; padding: 24px; }
  h1 { font-size: 1.4rem; font-weight: 500; margin-bottom: 16px; opacity: 0.7; letter-spacing: 1px; }
  #status { font-size: 5rem; font-weight: 700; padding: 24px 64px; border-radius: 16px;
            margin-bottom: 24px; transition: all 0.2s ease; }
  #status.off { background: #1a1a2e; color: #555; }
  #status.on  { background: #0a2e0a; color: #0f0; box-shadow: 0 0 40px rgba(0,255,0,0.2); }
  #label { font-size: 1.1rem; opacity: 0.5; margin-bottom: 24px; }
  img { max-width: 720px; width: 100%; border-radius: 12px; border: 2px solid #222; }
  #rec { display: none; color: #f44; font-weight: 600; margin-top: 12px; font-size: 1rem; }
  #rec.active { display: block; }
</style></head>
<body>
  <h1>YOLO CAMERA</h1>
  <div id="status" class="off">human = 0</div>
  <div id="label">detecting...</div>
  <img src="/stream" alt="camera">
  <div id="rec">REC</div>
<script>
  async function poll() {
    try {
      const r = await fetch('/api/status');
      const d = await r.json();
      const el = document.getElementById('status');
      el.textContent = 'human = ' + d.human;
      el.className = d.human ? 'on' : 'off';
      document.getElementById('label').textContent = d.detections + ' objects detected';
      const rec = document.getElementById('rec');
      rec.className = d.recording ? 'active' : '';
    } catch(e) {}
    setTimeout(poll, 200);
  }
  poll();
</script>
</body></html>"""

@app.route("/api/status")
def api_status():
    return jsonify({"human": int(human), "recording": recording,
                    "detections": len(latest_detections) if latest_detections else 0})

def mjpeg_gen():
    while running:
        frame = web_display_frame
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")
        time.sleep(0.05)

@app.route("/stream")
def stream():
    return Response(mjpeg_gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

def start_web():
    app.run(host="0.0.0.0", port=WEB_PORT, threaded=True)

web_thread = threading.Thread(target=start_web, daemon=True)
web_thread.start()

listener = threading.Thread(target=key_listener, daemon=True)
listener.start()
capturer = threading.Thread(target=capture_thread, daemon=True)
capturer.start()

print("Controls (type in this terminal):")
print("  R = start recording | S = stop recording | Q = quit")
print(f"  Web UI: http://0.0.0.0:{WEB_PORT}\n")

try:
    while running:
        with lock:
            frame = latest_frame
        if frame is None:
            time.sleep(0.01)
            continue

        results = model.predict(frame, conf=CONF, verbose=False, imgsz=320)
        boxes = results[0].boxes
        dets = []
        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                dets.append({
                    "box": boxes.xyxy[i].tolist(),
                    "conf": float(boxes.conf[i]),
                    "name": model.names[cls_id],
                    "color": (0, 140, 255),
                })
        human = any(d["name"] == "person" for d in dets)
        sys.stdout.write(f"\r  human = {int(human)}   ")
        sys.stdout.flush()

        with lock:
            latest_detections = dets

        display = draw_detections(frame, dets)
        if recording:
            cv2.circle(display, (40, 40), 12, (0, 0, 255), -1)
        scale = DISPLAY_WIDTH / display.shape[1]
        small = cv2.resize(display, (DISPLAY_WIDTH, int(display.shape[0] * scale)))
        web_display_frame = small
        cv2.imshow("YOLO", small)
        cv2.waitKey(1)

        action = pending_action
        pending_action = None

        if action == "start" and not recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            rec_final_path = os.path.join(OUTPUT_DIR, f"{timestamp}.mp4")
            rec_temp_path = os.path.join(OUTPUT_DIR, f"_temp_{timestamp}.mp4")
            with lock:
                writer = cv2.VideoWriter(rec_temp_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                         30, (actual_w, actual_h))
                rec_frame_count = 0
                rec_start_time = time.perf_counter()
                recording = True
            print("Recording started...")

        elif action == "stop" and recording:
            with lock:
                recording = False
                elapsed = time.perf_counter() - rec_start_time
                writer.release()
                writer = None
                count = rec_frame_count
            finalize_video(rec_temp_path, rec_final_path, count, elapsed)

        elif action == "quit":
            break

    if recording and writer:
        with lock:
            recording = False
            elapsed = time.perf_counter() - rec_start_time
            writer.release()
            writer = None
            count = rec_frame_count
        finalize_video(rec_temp_path, rec_final_path, count, elapsed)

finally:
    running = False
    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    cv2.destroyAllWindows()
    rpicam_proc.terminate()
    rpicam_proc.wait()
