from pathlib import Path
from types import SimpleNamespace
import yaml
from ultralytics import YOLO



def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d

def export_to_onnx(cfg):
    model = YOLO(cfg.model if hasattr(cfg, "model") else "yolo11s.pt")
    model.export(format="onnx", nms=False, export_train_head=True)


def start_export():
    file_path = 'ultralytics/cfg/default.yaml'
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    cfg=dict_to_namespace(config_dict)
    export_to_onnx(cfg)
    exported_model_path=cfg.model.replace('.pt', '.onnx')
    print(f"Model exported to ONNX: {exported_model_path}")
    return exported_model_path