import os
from code_loader.contract.datasetclasses import PreprocessResponse
from ultralytics.utils import callbacks as callbacks_ult
from ultralytics.data import  build_yolo_dataset
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.detect import DetectionValidator
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
    callbacks = callbacks_ult.get_default_callbacks()
    data = check_det_dataset(cfg.data, autodownload=True)
    n_samples=len(os.listdir(os.path.join(os.path.dirname(data[phase]),'labels',os.path.basename(data[phase])[:-4])))
    dataset = build_yolo_dataset(cfg, data[phase],n_samples , data, mode='val', stride=32)
    if phase == 'val':
        predictor = DetectionValidator(args=cfg, _callbacks=callbacks)
        predictor.data = data
        return dataset, n_samples, predictor
    # dataloader=build_dataloader(dataset, 1, 0, shuffle=False, rank=-1)
    return dataset, n_samples

def pre_process_dataloader(preprocessresponse:PreprocessResponse, idx):
    batch, predictor= preprocessresponse.data['dataloader'][idx], preprocessresponse.data['predictor']
    batch = predictor.preprocess(batch)
    imgs, clss, bboxes, batch_idxs= batch['img'], batch['cls'], batch['bboxes'], batch['batch_idx']
    return imgs.numpy(), clss.numpy(), bboxes.numpy(), batch_idxs.numpy()

def get_labels_mapping(cfg):
    return yaml_load(os.path.join(os.path.pardir, "cfg", "datasets", cfg.data))['names']