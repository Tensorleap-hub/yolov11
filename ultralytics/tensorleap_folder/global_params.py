from pathlib import Path
import os
import yaml
from types import SimpleNamespace
from ultralytics.utils import callbacks as callbacks_ult
from ultralytics.models.yolo.detect import DetectionValidator
from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_file


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
def get_criterion(model_path):
    from ultralytics import YOLO
    from ultralytics.utils import IterableSimpleNamespace
    model_base = YOLO(model_path) # TODO - make this part of the data path and read from there or load (maybe use ultralytics way)
    criterion = model_base.init_criterion()
    criterion.hyp = IterableSimpleNamespace(**criterion.hyp)
    criterion.hyp.box = 7.5
    criterion.hyp.cls = 0.5
    criterion.hyp.dfl = 1.5
    return criterion
def get_dataset_yaml(cfg):
    dataset_yaml_file=check_file(cfg.data)
    return  yaml_load(dataset_yaml_file, append_filename=True)

# def get_labels_mapping(cfg):
#     return yaml_load(os.path.join(os.path.pardir, "cfg", "datasets", cfg.data))['names']

def get_predictor_obj(cfg,yolo_data):
    callbacks = callbacks_ult.get_default_callbacks()
    predictor = DetectionValidator(args=cfg, _callbacks=callbacks)
    predictor.data = yolo_data
    predictor.end2end = False
    return predictor

root = Path(__file__).resolve().parent.parent
file_path = os.path.join(root, 'cfg/default.yaml')
with open(file_path, 'r') as file:
    config_dict = yaml.safe_load(file)

cfg = dict_to_namespace(config_dict)
yolo_data=get_yolo_data(cfg)
dataset_yaml=get_dataset_yaml(cfg)
criterion=get_criterion(Path(dataset_yaml['path'])/cfg.model)
all_clss=dataset_yaml["names"]
predictor=get_predictor_obj(cfg,yolo_data)