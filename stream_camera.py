import cv2
import os
import threading
import time
from datetime import datetime
from flask import Flask, Response, jsonify
from ultralytics import YOLO

MODEL = "yolo26s.pt"
CAMERA = 0
CONF = 0.25
HOST = "0.0.0.0"
PORT = 5000
JPEG_QUALITY = 80
OUTPUT_DIR = "video_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)

app = Flask(__name__)
model = YOLO(MODEL)

latest_frame = None
latest_annotated = None
new_frame_event = threading.Event()

recording = False
video_writer = None
record_lock = threading.Lock()


def inference_loop():
    global latest_frame, latest_annotated, recording, video_writer
    cap = cv2.VideoCapture(CAMERA, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps} FPS")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            continue
        results = model.predict(frame, conf=CONF, verbose=False, imgsz=640)
        annotated = results[0].plot()

        with record_lock:
            if recording and video_writer is not None:
                video_writer.write(annotated)

        _, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        latest_frame = buf.tobytes()
        new_frame_event.set()
    cap.release()


def generate_frames():
    while True:
        new_frame_event.wait()
        new_frame_event.clear()
        frame = latest_frame
        if frame is None:
            continue
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@app.route("/")
def index():
    return """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>YOLO Stream</title>
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:#000; overflow:hidden; font-family:system-ui,sans-serif; }
  img { width:100vw; height:100vh; object-fit:contain; }
  #status {
    position:fixed; top:20px; right:20px; padding:10px 20px;
    border-radius:8px; font-size:14px; font-weight:600;
    color:#fff; z-index:10; transition:all 0.3s;
  }
  #status.idle { background:rgba(255,255,255,0.15); }
  #status.rec  { background:rgba(220,38,38,0.85); }
  #help {
    position:fixed; bottom:20px; left:50%; transform:translateX(-50%);
    padding:8px 16px; border-radius:8px; background:rgba(255,255,255,0.12);
    color:rgba(255,255,255,0.7); font-size:13px; z-index:10;
  }
</style>
</head>
<body>
<div id="status" class="idle">IDLE</div>
<div id="help">Press <b>R</b> to record &bull; <b>S</b> to stop</div>
<img src="/stream">
<script>
const status = document.getElementById('status');
document.addEventListener('keydown', async (e) => {
  if (e.key === 'r' || e.key === 'R') {
    const res = await fetch('/record/start', {method:'POST'});
    const data = await res.json();
    if (data.recording) { status.textContent = '⏺ REC'; status.className = 'rec'; }
  }
  if (e.key === 's' || e.key === 'S') {
    const res = await fetch('/record/stop', {method:'POST'});
    const data = await res.json();
    if (!data.recording) { status.textContent = 'SAVED: ' + data.file; status.className = 'idle'; }
  }
});
</script>
</body></html>"""


@app.route("/stream")
def stream():
    return Response(generate_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame",
                    headers={"Cache-Control": "no-cache, no-store, must-revalidate",
                             "Pragma": "no-cache",
                             "Expires": "0",
                             "Connection": "close"})


@app.route("/record/start", methods=["POST"])
def record_start():
    global recording, video_writer
    with record_lock:
        if not recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}.mp4"
            filepath = os.path.join(OUTPUT_DIR, filename)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(filepath, fourcc, 30, (1920, 1080))
            recording = True
            print(f"Recording started: {filepath}")
            return jsonify(recording=True, file=filename)
        return jsonify(recording=True, file="already recording")


@app.route("/record/stop", methods=["POST"])
def record_stop():
    global recording, video_writer
    with record_lock:
        if recording:
            recording = False
            video_writer.release()
            video_writer = None
            print("Recording stopped and saved")
            return jsonify(recording=False, file="saved")
        return jsonify(recording=False, file="not recording")


if __name__ == "__main__":
    threading.Thread(target=inference_loop, daemon=True).start()
    time.sleep(2)
    print(f"Stream at http://127.0.0.1:{PORT}")
    print("Press R in browser to record, S to stop")
    app.run(host=HOST, port=PORT, threaded=True)
