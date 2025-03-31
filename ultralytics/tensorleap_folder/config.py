from pathlib import Path
import os
import yaml
from types import SimpleNamespace


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d
def get_yolo_data(cfg):
    from ultralytics.data.utils import check_det_dataset
    return check_det_dataset(cfg.data, autodownload=True)
def get_criterion():
    from ultralytics import YOLO
    from ultralytics.utils import IterableSimpleNamespace
    model_base = YOLO("yolo11s.pt") # TODO - make this part of the data path and read from there or load (maybe use ultralytics way)
    criterion = model_base.init_criterion()
    criterion.hyp = IterableSimpleNamespace(**criterion.hyp)
    criterion.hyp.box = 7.5
    criterion.hyp.cls = 0.5
    criterion.hyp.dfl = 1.5
    return criterion

root = Path(__file__).resolve().parent.parent
file_path = os.path.join(root, 'cfg/default.yaml')
with open(file_path, 'r') as file:
    config_dict = yaml.safe_load(file)

cfg = dict_to_namespace(config_dict)
yolo_data=get_yolo_data(cfg)
criterion=get_criterion()