from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss
from ultralytics.tensorleap_folder.config import cfg
from ultralytics.tensorleap_folder.utils import create_data_with_ult,pre_process_dataloader
from typing import List
import numpy as np
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType
from code_loader.contract.enums import LeapDataType
from code_loader.visualizers.default_visualizers import LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_gt_encoder,
                                                                 tensorleap_input_encoder, tensorleap_metadata,
                                                                 tensorleap_custom_visualizer)



@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    data_loader_val,n_sampels_val, predictor=create_data_with_ult(cfg,phase='val')
    data_loader_train,n_sampels_train=create_data_with_ult(cfg,phase='train')

    # imgs_train, clss_train, bboxes_train, batch_idxs_train=create_data_with_ult(cfg,phase='train')
    val = PreprocessResponse(sample_ids=np.arange(n_sampels_val), data={'dataloader':data_loader_val, 'predictor':predictor},sample_id_type=int)
    train = PreprocessResponse(sample_ids=np.arange(n_sampels_train), data={'dataloader':data_loader_train, 'predictor':predictor},sample_id_type=int, state=DataStateType.training)
    response = [val,train]
    return response

# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image')
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    imgs, _, _, _ =pre_process_dataloader(preprocess, idx)

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
    _, clss, bboxes, _=pre_process_dataloader(preprocessing, idx)
    return np.concatenate([bboxes,clss],axis=1)


# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


@tensorleap_custom_loss("dummy_loss")
def dummy_loss(pred,gt):
    return np.zeros(1)

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

    # val_smps_plot_tf = plot_images(torch.tensor(image).unsqueeze(0), torch.tensor(np.repeat(batch_idx,bb_gt.shape[0])), bb_gt[:,-1].astype(int), torch.tensor(bb_gt[:,:-1]), names=get_labels_mapping(cfg), save=False, threaded=False)
    # bb_object: List[BoundingBox] = bb_array_to_object(bb_gt, iscornercoded=False, bg_label=CONFIG['BACKGROUND_LABEL'],
    #                                                   is_gt=True)
    # bb_object = [bbox for bbox in bb_object if bbox.label in CATEGORIES_no_background]
    #
    # return LeapImageWithBBox(data=(image.transpose(1,2,0)), bounding_boxes=bb_gt)
    pass

@tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:

    return LeapImage((image.transpose(1,2,0)), compress=False)

@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, predictions: np.ndarray) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    pass
    return LeapImageWithBBox(data=(image * 255).astype(np.uint8), bounding_boxes=predictions)

if __name__ == '__main__':
    leap_binder.check()
