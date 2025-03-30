import torch
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss
from ultralytics.tensorleap_folder.config import cfg, yolo_data
from ultralytics.tensorleap_folder.utils import create_data_with_ult, pre_process_dataloader, get_labels_mapping, \
    get_predictor_obj
from typing import List
import numpy as np
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType
from code_loader.contract.enums import LeapDataType
from code_loader.visualizers.default_visualizers import LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_gt_encoder,
                                                                 tensorleap_input_encoder, tensorleap_metadata,
                                                                 tensorleap_custom_visualizer)

from ultralytics.utils import yaml_load
from ultralytics.utils.checks import check_file

from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.utils import rescale_min_max
from ultralytics.utils.plotting import output_to_target


@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    data_loader_val,n_sampels_val=create_data_with_ult(cfg,yolo_data,phase='val')
    data_loader_train,n_sampels_train=create_data_with_ult(cfg,yolo_data,phase='train')

    val = PreprocessResponse(sample_ids=list(range(n_sampels_val)), data={'dataloader':data_loader_val},sample_id_type=int, state=DataStateType.validation)
    train = PreprocessResponse(sample_ids=list(range(n_sampels_train)), data={'dataloader':data_loader_train},sample_id_type=int, state=DataStateType.training)
    response = [val,train]
    return response

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image')
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    predictor = get_predictor_obj(cfg,yolo_data)
    imgs, _, _, _ =pre_process_dataloader(preprocess, idx, predictor)

    return imgs.astype('float32')


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    """
        Description: This function takes an integer index idx and a PreprocessResponse object data as input and returns an
                     array of bounding boxes and label per bbox [x_center, y_center, width, height, label] representing ground truth annotations.

        Input: idx (int): sample index.
        data (PreprocessResponse): An object of type PreprocessResponse containing data attributes.
        Output: bounding_boxes (np.ndarray): An array of bounding boxes extracted from the instance segmentation polygons in
                the JSON data. Each bounding box is represented as an array containing [x_center, y_center, width, height, label].
        """
    predictor = get_predictor_obj(cfg,yolo_data)
    _, clss, bboxes, _=pre_process_dataloader(preprocessing, idx,predictor)
    return np.concatenate([bboxes,clss],axis=1)


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


@tensorleap_custom_loss("dummy_loss")
def dummy_loss(pred,gt):
    return np.random.rand(1)

@tensorleap_custom_visualizer("bb_gt_decoder", LeapDataType.ImageWithBBox)
def gt_bb_decoder(image: np.ndarray, bb_gt: np.ndarray) -> LeapImageWithBBox:
    """
    This function overlays ground truth bounding boxes (BBs) on the input image.

    Parameters:
    image (np.ndarray): The input image for which the ground truth bounding boxes need to be overlaid.
    bb_gt (np.ndarray): The ground truth bounding box array for the input image.

    Returns:
    An instance of LeapImageWithBBox containing the input image with ground truth bounding boxes overlaid.
    """
    dataset_yaml_file=check_file(cfg.data)
    all_clss = yaml_load(dataset_yaml_file, append_filename=True)['names']
    bbox = [BoundingBox(x=bbx[0], y=bbx[1], width=bbx[2], height=bbx[3], confidence=1, label=all_clss.get(int(bbx[4]),'Unknown Class')) for bbx in bb_gt]
    image = rescale_min_max(image)
    return LeapImageWithBBox(data=(image.transpose(1,2,0)), bounding_boxes=bbox)

@tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    image = rescale_min_max(image)
    return LeapImage((image.transpose(1,2,0)), compress=False)

@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, predictions: np.ndarray) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    dataset_yaml_file=check_file(cfg.data)
    all_clss = yaml_load(dataset_yaml_file, append_filename=True)['names']
    predictor = get_predictor_obj(cfg,yolo_data)
    y_pred = predictor.postprocess(torch.from_numpy(predictions))
    _, cls_temp, bbx_temp, conf_temp = output_to_target(y_pred, max_det=predictor.args.max_det)
    t_pred = np.concatenate([bbx_temp, np.expand_dims(conf_temp, 1), np.expand_dims(cls_temp, 1)], axis=1)
    post_proc_pred = t_pred[t_pred[:, 4] >  (getattr(cfg, "conf", 0.25) or 0.25)]
    post_proc_pred[:, :4:2] /= image.shape[1]
    post_proc_pred[:, 1:4:2] /= image.shape[2]
    bbox = [BoundingBox(x=bbx[0], y=bbx[1], width=bbx[2], height=bbx[3], confidence=bbx[4], label=all_clss.get(int(bbx[5]),'Unknown Class')) for bbx in post_proc_pred]

    return LeapImageWithBBox(data=(image.transpose(1,2,0)), bounding_boxes=bbox)

if __name__ == '__main__':
    leap_binder.check()
