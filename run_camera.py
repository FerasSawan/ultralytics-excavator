from ultralytics import YOLO

MODEL = "yolo26m.pt"  # change this to test different models
CAMERA = 0            # camera index, try 1 if Innomaker isn't 0
CONF = 0.25           # minimum confidence threshold

model = YOLO(MODEL)
for result in model.predict(source=CAMERA, show=True, stream=True, conf=CONF):
    pass