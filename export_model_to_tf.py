from ultralytics import YOLO
model = YOLO("yolo5s.pt")
model.export(format="onnx", nms=False, export_train_head=True)

