import os
from code_loader.contract.datasetclasses import PreprocessResponse
from ultralytics.data import  build_yolo_dataset


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

def create_data_with_ult(cfg,yolo_data, phase='val'):
    n_samples=len(os.listdir(os.path.join(os.path.dirname(yolo_data[phase]),'labels',os.path.basename(yolo_data[phase])[:-4])))
    dataset = build_yolo_dataset(cfg, yolo_data[phase],n_samples , yolo_data, mode='val', stride=32)
    return dataset, n_samples

def pre_process_dataloader(preprocessresponse:PreprocessResponse, idx, predictor):
    batch= preprocessresponse.data['dataloader'][idx]
    batch = predictor.preprocess(batch)
    imgs, clss, bboxes, batch_idxs= batch['img'], batch['cls'], batch['bboxes'], batch['batch_idx']
    return imgs.numpy(), clss.numpy(), bboxes.numpy(), batch_idxs.numpy()
