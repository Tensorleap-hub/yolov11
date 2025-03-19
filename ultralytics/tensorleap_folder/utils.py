import os
from ultralytics.data import  build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.build import build_dataloader
from ultralytics.utils.__init__ import yaml_load

def metadata_label(digit_int) -> int:
    return digit_int


def metadata_even_odd(digit_int) -> str:
    if digit_int % 2 == 0:
        return "even"
    else:
        return "odd"


def metadata_circle(digit_int) -> str:
    if digit_int in [0, 6, 8, 9]:
        return 'yes'
    else:
        return 'no'


def create_data_with_ult(cfg,phase='val'):

    data = check_det_dataset(cfg.data, autodownload=True)
    dataset = build_yolo_dataset(cfg, data['path'], 1, data, mode=phase, stride=32)
    dataloader=build_dataloader(dataset, len(dataset), 0, shuffle=False, rank=-1)
    batch = next(iter(dataloader))
    imgs, clss, bboxes, batch_idxs= batch['img'], batch['cls'], batch['bboxes'], batch['batch_idx']
    return imgs.numpy(), clss.numpy(), bboxes.numpy(), batch_idxs.numpy()

def get_labels_mapping(cfg):
    return yaml_load(os.path.join(os.path.pardir, "cfg", "datasets", cfg.data))['names']