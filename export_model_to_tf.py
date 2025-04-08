from ultralytics import YOLO
model = YOLO("yolo11s.pt")
model.export(format="onnx", nms=False, export_train_head=True)

