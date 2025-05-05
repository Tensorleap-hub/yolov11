import os
import re
import shutil
from pathlib import Path

import numpy as np
import torch
from code_loader.contract.datasetclasses import PreprocessResponse
from ultralytics.data import  build_yolo_dataset
from ultralytics.utils.plotting import output_to_target


def create_data_with_ult(cfg,yolo_data, phase='val'):
    n_samples = len(os.listdir(yolo_data[phase]))
    dataset = build_yolo_dataset(cfg, yolo_data[phase],n_samples , yolo_data, mode='val', stride=32)
    return dataset, n_samples

def pre_process_dataloader(preprocessresponse:PreprocessResponse, idx, predictor):
    batch= preprocessresponse.data['dataloader'][idx]
    batch = predictor.preprocess(batch)
    imgs, clss, bboxes, batch_idxs, ori_shape, resized_shape,ratio_pad = batch['img'], batch['cls'], batch['bboxes'], batch['batch_idx'],batch['ori_shape'],batch['resized_shape'],batch['ratio_pad']
    return imgs.numpy(), clss.numpy(), bboxes.numpy(), batch_idxs.numpy()


def pred_post_process(y_pred, predictor, image, cfg):
    y_pred = predictor.postprocess(torch.from_numpy(y_pred).unsqueeze(0))
    _, cls_temp, bbx_temp, conf_temp = output_to_target(y_pred, max_det=predictor.args.max_det)
    t_pred = np.concatenate([bbx_temp, np.expand_dims(conf_temp, 1), np.expand_dims(cls_temp, 1)], axis=1)
    post_proc_pred = t_pred[t_pred[:, 4] > (getattr(cfg, "conf", 0.3) or 0.3)]
    post_proc_pred[:, :4:2] /= image.shape[1]
    post_proc_pred[:, 1:4:2] /= image.shape[2]
    return post_proc_pred


def extract_mapping(m_path):
    def extract_yolo_variant(filename):
        pattern = r'yolo(?:v)?\d+[a-zA-Z]'
        match = re.search(pattern, filename)
        if not match:
            return False
        else:
            return f"{match.group()}"[:-1].replace('v','')

    filename=Path(m_path).stem
    model_type=extract_yolo_variant(filename)
    root = Path.cwd()
    mapping_folder_path =root / Path('ultralytics/tensorleap_folder/mapping')
    source_file = mapping_folder_path / f'leap_mapping_{model_type}.yaml'

    if not model_type or not os.path.exists(source_file):
        print(f"No Mapping for {m_path} was found, put your mapping in the root directory.")
    else:
        destination_file = root/ 'leap_mapping.yaml'
        shutil.copy(source_file, destination_file)
        print(f"Extracting mapping for {model_type} completed")

