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
def get_criterion(model_path,cfg):
    from ultralytics import YOLO
    from ultralytics.utils import IterableSimpleNamespace
    if not model_path.is_absolute():
        model_path = (cfg.tensorleap_path / model_path).resolve()
    assert model_path.is_relative_to(cfg.tensorleap_path), (
        f"❌ {model_path!r} is not inside tensorleap path {cfg.tensorleap_path!r}" )
    model_base = YOLO(model_path)
    criterion = model_base.init_criterion()
    criterion.hyp = IterableSimpleNamespace(**criterion.hyp)
    criterion.hyp.box = cfg.box
    criterion.hyp.cls = cfg.box
    criterion.hyp.dfl = cfg.box

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
criterion=get_criterion(Path(cfg.model),cfg)
all_clss=dataset_yaml["names"]
predictor=get_predictor_obj(cfg,yolo_data)