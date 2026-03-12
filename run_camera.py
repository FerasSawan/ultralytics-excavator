import cv2
import keyboard
import os
import time
from datetime import datetime
from ultralytics import YOLO

MODEL = "yolo26m.pt"  # change this to test different models
CAMERA = 0            # camera index, try 1 if Innomaker isn't 0
CONF = 0.25           # minimum confidence threshold
OUTPUT_DIR = "video_outputs"

os.makedirs(OUTPUT_DIR, exist_ok=True)
model = YOLO(MODEL)

cap = cv2.VideoCapture(CAMERA)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

recording = False
writer = None
rec_frame_count = 0
rec_start_time = 0.0
rec_temp_path = ""
rec_final_path = ""
pending_action = None


def on_r():
    global pending_action
    pending_action = "start"

def on_s():
    global pending_action
    pending_action = "stop"

def on_q():
    global pending_action
    pending_action = "quit"


keyboard.add_hotkey("r", on_r, suppress=False)
keyboard.add_hotkey("s", on_s, suppress=False)
keyboard.add_hotkey("q", on_q, suppress=False)

print("Global hotkeys active (works from any window):")
print("  R = start recording | S = stop recording | Q = quit")


def finalize_video(temp_path, final_path, frame_count, elapsed):
    """Re-mux the temp video at the true measured FPS so playback = real time."""
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


while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        break

    results = model.predict(frame, conf=CONF, verbose=False, imgsz=640)
    annotated = results[0].plot()

    if recording:
        writer.write(annotated)
        rec_frame_count += 1
        cv2.circle(annotated, (40, 40), 12, (0, 0, 255), -1)

    cv2.imshow("YOLO", annotated)
    cv2.waitKey(1)

    action = pending_action
    pending_action = None

    if action == "start" and not recording:
        h, w = annotated.shape[:2]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        rec_final_path = os.path.join(OUTPUT_DIR, f"{timestamp}.mp4")
        rec_temp_path = os.path.join(OUTPUT_DIR, f"_temp_{timestamp}.mp4")
        writer = cv2.VideoWriter(rec_temp_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
        rec_frame_count = 0
        rec_start_time = time.perf_counter()
        recording = True
        print("Recording started...")

    elif action == "stop" and recording:
        elapsed = time.perf_counter() - rec_start_time
        recording = False
        writer.release()
        writer = None
        finalize_video(rec_temp_path, rec_final_path, rec_frame_count, elapsed)

    elif action == "quit":
        break

if recording and writer:
    elapsed = time.perf_counter() - rec_start_time
    writer.release()
    finalize_video(rec_temp_path, rec_final_path, rec_frame_count, elapsed)

cap.release()
cv2.destroyAllWindows()
keyboard.unhook_all()
