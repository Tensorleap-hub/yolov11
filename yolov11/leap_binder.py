from typing import Dict, List, Union
import numpy.typing as npt
import numpy as np
from code_loader import leap_binder
from code_loader.visualizers.default_visualizers import LeapHorizontalBar, LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (
    tensorleap_preprocess,
    tensorleap_gt_encoder,
    tensorleap_input_encoder,
    tensorleap_metadata,
    tensorleap_custom_loss)
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType

#ultralytics imports
from coremltools.converters.mil.testing_reqs import tf
from matplotlib import pyplot as plt
from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.models.yolo.model import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.converter import convert_coco
from ultralytics.data.build import build_dataloader
from ultralytics.utils import callbacks
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.tensorleap_folder.global_params import cfg

@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    data = check_det_dataset(cfg.data, autodownload=True)
    val_dataset = build_yolo_dataset(cfg, data['path'], 1, data, mode='val', stride=32)
    train_dataset = build_yolo_dataset(cfg, data['path'], 1, data, mode='train', stride=32)

    val_dataloader = build_dataloader(val_dataset, 1, 0, shuffle=False,
                                  rank=-1)  # build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)
    train_dataloader = build_dataloader(train_dataset, 1, 0, shuffle=False,
                                  rank=-1)
    train = PreprocessResponse(sample_ids=range(len(train_dataloader)), data=train_dataloader, sample_id_type=int, state=DataStateType.training)
    val = PreprocessResponse(sample_ids=range(len(val_dataloader)), data=val_dataloader, sample_id_type=int)
    # leap_binder.cache_container["classes_avg_images"] = calc_classes_centroid(train_X, train_Y)
    response = [train, val]
    return response

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image')
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    return preprocess.data['images'][idx].astype('float32')


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_gt_encoder('classes')
def gt_encoder(idx: int, preprocessing: PreprocessResponse) -> np.ndarray:
    return preprocessing.data['labels'][idx].astype('float32')


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


@tensorleap_custom_loss("dummy_loss")
def dummy_loss(pred,gt):
    return np.zeros(1)

#
# # Metadata functions allow to add extra data for a later use in analysis.
# # This metadata adds the int digit of each sample (not a hot vector).
# @tensorleap_metadata('metadata_one_hot_digit')
# def metadata_one_hot_digit(idx: int, preprocess: PreprocessResponse) -> Dict[str, Union[str, int]]:
#     one_hot_digit = gt_encoder(idx, preprocess)
#     digit = one_hot_digit.argmax()
#     digit_int = int(digit)
#
#     res = {
#         'label':metadata_label(digit_int),
#         'even_odd': metadata_even_odd(digit_int),
#         'circle': metadata_circle(digit_int)
#     }
#     return res
#
#
# @tensorleap_metadata('euclidean_diff_from_class_centroid')
# def metadata_euclidean_distance_from_class_centroid(idx: int,
#                                                     preprocess: Union[PreprocessResponse, list]) -> np.ndarray:
#     # ### calculate euclidean distance from the average image of the specific class
#     # sample_input = preprocess.data['images'][idx]
#     # label = preprocess.data['labels'][idx]
#     # label = str(np.argmax(label))
#     # class_average_image = leap_binder.cache_container["classes_avg_images"][label]
#     return 1
#
#
# @tensorleap_custom_visualizer('horizontal_bar_classes', LeapHorizontalBar.type)
# def combined_bar(data: NDArray[float], gt:NDArray[float]) -> LeapHorizontalBar:
#     return LeapHorizontalBar(body=data, gt=gt, labels=CONFIG['names'])
#
#
# @tensorleap_custom_metric('metrics', direction=MetricDirection.Upward)
# def metrics(output_pred: NDArray[float]) -> Dict[str, NDArray[Union[float, int]]]:
#     prob = output_pred.max(axis=-1)
#     pred_idx = output_pred.argmax(axis=-1)
#     metrics_dict = {'prob': prob,
#                     'prd_idx': pred_idx}
#     return metrics_dict
#
# @tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
# def image_visualizer(image: npt.NDArray[np.float32]) -> LeapImage:
#     # TODO: Revert the image normalization if needed
#     return LeapImage((image*255).astype(np.uint8), compress=False)

# Adding a name to the prediction, and supplying it with label names.
# leap_binder.add_prediction(name='classes', labels=CONFIG['names'], channel_dim=-1)

if __name__ == '__main__':
    leap_binder.check()
