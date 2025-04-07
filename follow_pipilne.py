from ultralytics import YOLO

# Load a model
model = YOLO("/users/yamtawachi/tensorleap/datasets/yolo11n.pt")

# Customize validation settings
validation_results = model.val(data="coco8.yaml", imgsz=640, batch=1, conf=0.25, iou=0.6, device="cpu")