import torch
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss, tensorleap_custom_metric, \
    tensorleap_instances_length_encoder, tensorleap_custom_instances_metric
from pygments.formatters import img
from tensorflow.python.ops.numpy_ops.np_array_ops import ones_like

from ultralytics.tensorleap_folder.global_params import cfg, yolo_data, criterion, all_clss, predictor
from ultralytics.tensorleap_folder.utils import create_data_with_ult, pre_process_dataloader
from typing import List, Dict, Union
import numpy as np
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType, SamplePreprocessResponse, ElementInstance
from code_loader.contract.enums import LeapDataType, MetricDirection
from code_loader.visualizers.default_visualizers import LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_element_instance_preprocess, tensorleap_gt_encoder,
                                                                 tensorleap_input_encoder, tensorleap_metadata,
                                                                 tensorleap_custom_visualizer, tensorleap_instances_masks_encoder)
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.utils import rescale_min_max
from ultralytics.utils.plotting import output_to_target
from ultralytics.utils.metrics import box_iou
import cv2


COCO_ID_TO_NAME = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    12: "stop sign",
    13: "parking meter",
    14: "bench",
    15: "bird",
    16: "cat",
    17: "dog",
    18: "horse",
    19: "sheep",
    20: "cow",
    21: "elephant",
    22: "bear",
    23: "zebra",
    24: "giraffe",
    25: "backpack",
    26: "umbrella",
    27: "handbag",
    28: "tie",
    29: "suitcase",
    30: "frisbee",
    31: "skis",
    32: "snowboard",
    33: "sports ball",
    34: "kite",
    35: "baseball bat",
    36: "baseball glove",
    37: "skateboard",
    38: "surfboard",
    39: "tennis racket",
    40: "bottle",
    41: "wine glass",
    42: "cup",
    43: "fork",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    48: "apple",
    49: "sandwich",
    50: "orange",
    51: "broccoli",
    52: "carrot",
    53: "hot dog",
    54: "pizza",
    55: "donut",
    56: "cake",
    57: "chair",
    58: "couch",
    59: "potted plant",
    60: "bed",
    61: "dining table",
    62: "toilet",
    63: "tv",
    64: "laptop",
    65: "mouse",
    66: "remote",
    67: "keyboard",
    68: "cell phone",
    69: "microwave",
    70: "oven",
    71: "toaster",
    72: "sink",
    73: "refrigerator",
    74: "book",
    75: "clock",
    76: "vase",
    77: "scissors",
    78: "teddy bear",
    79: "hair drier",
    80: "toothbrush"
}

def save_chw_image(img: np.ndarray, path: str):
    imgg = rescale_min_max(img.copy()).transpose(1, 2, 0)
    img_bgr = cv2.cvtColor(imgg, cv2.COLOR_RGB2BGR)  # Convert RGB → BGR for OpenCV
    cv2.imwrite(path, img_bgr)

# ----------------------------------------------------data processing---------------------------------------------------
@tensorleap_instances_masks_encoder('image')
def instance_mask_encoder(idx: str, preprocess: PreprocessResponse, instance_idx) -> ElementInstance:
    gt = gt_encoder(idx, preprocess)
    label = gt[instance_idx]
    mask = np.zeros((3, 640, 640))
    x, y, w, h, label_id = label
    if np.isnan([x, y, w, h, label_id]).any():
        return None
    img_width, img_height = mask.shape[1], mask.shape[2]
    x, y, w, h = round(x * img_width - ((w * img_width) / 2)), round(y * img_height - ((h * img_height) / 2)), round(w * img_width), round(h * img_height)

    mask[:, y:y+h, x:x+w] = 1

    element_instance = ElementInstance(COCO_ID_TO_NAME[int(label_id) + 1], mask)

    return element_instance


@tensorleap_instances_length_encoder('image')
def instances_length_encoder(idx: str, preprocess: PreprocessResponse) -> int:
    gt = gt_encoder(idx, preprocess)
    for label in gt:
        x, y, w, h, label_id = label
        if np.isnan([x, y, w, h, label_id]).any():
            return 0
    return len(gt)

@tensorleap_element_instance_preprocess(instances_length_encoder, instance_mask_encoder)
# @tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    dataset_types = [DataStateType.training, DataStateType.validation]
    phases = ['train', 'val']
    responses = []
    if cfg.tensorleap_use_test:
        phases.append('test')
        dataset_types.append(DataStateType.test)
    if cfg.tensorleap_use_unlabeled:
        phases.append('unlabeled')
        dataset_types.append(DataStateType.unlabeled)
    for i, (phase, dataset_type) in enumerate(zip(phases, dataset_types)):
        data_loader, n_samples = create_data_with_ult(cfg, yolo_data, phase=phase)
        responses.append(
            PreprocessResponse(sample_ids=[str(idd + i * 1000) for idd in range(50)],
                               data={'dataloader':data_loader},
                               state=dataset_type))

    return responses


# ------------------------------------------input and gt----------------------------------------------------------------


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image',channel_dim=1)
def input_encoder(idx: str, preprocess: PreprocessResponse) -> np.ndarray:
    imgs, _, _,_=pre_process_dataloader(preprocess, int(idx), predictor)

    return imgs.astype('float32')


# Ground truth encoder fetches the label with the index `idx` from the `labels` array set in
# the PreprocessResponse's data. Returns a numpy array containing a hot vector label correlated with the sample.
@tensorleap_gt_encoder('classes')
def gt_encoder(idx: str, preprocessing: PreprocessResponse) -> np.ndarray:
    """
        Description: This function takes an integer index idx and a PreprocessResponse object data as input and returns an
                     array of bounding boxes and label per bbox [x_center, y_center, width, height, label] representing ground truth annotations.

        Input: idx (int): sample index.
        data (PreprocessResponse): An object of type PreprocessResponse containing data attributes.
        Output: bounding_boxes (np.ndarray): An array of bounding boxes extracted from the instance segmentation polygons in
                the JSON data. Each bounding box is represented as an array containing [x_center, y_center, width, height, label].
        """
    _, clss, bboxes, _ =pre_process_dataloader(preprocessing, int(idx),predictor)
    if clss.shape[0]==0 and  bboxes.shape[0]==0:
        return np.full((1, 5), np.nan,dtype=np.float32)
    elif clss.shape[0]==0:
        temp_array=np.full((bboxes.shape[0], 5), np.nan,dtype=np.float32)
        temp_array[:,:4]=bboxes
        return temp_array
    elif bboxes.shape[0]==0:
        temp_array = np.full((clss.shape[0], 5), np.nan,dtype=np.float32)
        temp_array[:, 4] = clss
        return temp_array
    return np.concatenate([bboxes,clss],axis=1)

# ----------------------------------------------------------metadata----------------------------------------------------

# Metadata functions allow to add extra data for a later use in analysis.
# This metadata adds the int digit of each sample (not a hot vector).
@tensorleap_metadata('metadata_sample_index')
def metadata_sample_index(idx: str, preprocess: PreprocessResponse) -> str:
    return idx


@tensorleap_metadata("image info")
def misc_metadata(idx: str, data: PreprocessResponse) -> Dict[str, Union[str, int]]:
    idx_int = int(idx)
    clss_info=np.unique(data.data['dataloader'].labels[idx_int]["cls"],return_counts=True)
    d = {
        "image path": data.data['dataloader'].im_files[idx_int],
        "target path": data.data['dataloader'].label_files[idx_int],
        "bbox_format": data.data['dataloader'].labels[idx_int]["bbox_format"],
        "normalized image": data.data['dataloader'].labels[idx_int]["normalized"],
        "idx":idx,
        "# unique classes" : len(clss_info[0]),
        "# of objects": clss_info[1].sum(),
     }
    return d

# ----------------------------------------------------------loss--------------------------------------------------------

@tensorleap_custom_loss("total_loss")
def loss(pred80,pred40,pred20,gt,demo_pred):
    gt=np.squeeze(gt,axis=0)
    d={}
    d["bboxes"] = torch.from_numpy(gt[...,:4])
    d["cls"] = torch.from_numpy(gt[...,4])
    d["batch_idx"] = torch.zeros_like(d['cls'])
    y_pred_torch = [torch.from_numpy(s) for s in [pred80,pred40,pred20]]
    all_loss,_= criterion(y_pred_torch, d)
    return all_loss.unsqueeze(0).numpy()


# ------------------------------------------------------visualizers-----------------------------------------------------
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
    bbox = [BoundingBox(x=bbx[0], y=bbx[1], width=bbx[2], height=bbx[3], confidence=1, label=all_clss.get(int(bbx[4]) if not np.isnan(bbx[4]) else -1, 'Unknown Class')) for bbx in bb_gt.squeeze(0)]
    image = rescale_min_max(image.squeeze(0))
    return LeapImageWithBBox(data=(image.transpose(1,2,0)), bounding_boxes=bbox)

@tensorleap_custom_visualizer('image_visualizer', LeapDataType.Image)
def image_visualizer(image: np.ndarray) -> LeapImage:
    image = rescale_min_max(image.squeeze(0))
    return LeapImage((image.transpose(1,2,0)), compress=False)

@tensorleap_custom_visualizer('image_visualizer_original', LeapDataType.Image)
def image_visualizer_original(image: np.ndarray, sample_preprocess_response: SamplePreprocessResponse):
    id = sample_preprocess_response.sample_ids[0]
    sample_id = sample_preprocess_response.preprocess_response.instance_to_sample_ids_mappings[str(id)]
    image = input_encoder(sample_id, sample_preprocess_response.preprocess_response)
    image = rescale_min_max(image)
    return LeapImage((image.transpose(1,2,0)), compress=False)

@tensorleap_custom_visualizer("bb_decoder", LeapDataType.ImageWithBBox)
def bb_decoder(image: np.ndarray, predictions: np.ndarray) -> LeapImageWithBBox:
    """
    Overlays the BB predictions on the image
    """
    image=image.squeeze(0)
    y_pred = predictor.postprocess(torch.from_numpy(predictions))
    _, cls_temp, bbx_temp, conf_temp = output_to_target(y_pred, max_det=predictor.args.max_det)
    t_pred = np.concatenate([bbx_temp, np.expand_dims(conf_temp, 1), np.expand_dims(cls_temp, 1)], axis=1)
    post_proc_pred = t_pred[t_pred[:, 4] >  (getattr(cfg, "conf", 0.25) or 0.25)]
    post_proc_pred[:, :4:2] /= image.shape[1]
    post_proc_pred[:, 1:4:2] /= image.shape[2]
    bbox = [BoundingBox(x=bbx[0], y=bbx[1], width=bbx[2], height=bbx[3], confidence=bbx[4], label=all_clss.get(int(bbx[5]),'Unknown Class')) for bbx in post_proc_pred]
    image = rescale_min_max(image)
    return LeapImageWithBBox(data=(image.transpose(1,2,0)), bounding_boxes=bbox)





# ---------------------------------------------------------metrics------------------------------------------------------
@tensorleap_custom_metric("ious", direction=MetricDirection.Upward)
def iou_dic(y_pred: np.ndarray, preprocess: SamplePreprocessResponse): #-> Dict[str, Union[float, int]]:
    batch=preprocess.preprocess_response.data['dataloader'][int(preprocess.sample_ids)]
    batch["imgsz"]=(batch["resized_shape"],)
    batch["ori_shape"]=(batch["ori_shape"],)
    batch["ratio_pad"]= (batch["ratio_pad"],)
    batch["img"]=batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred))[0]
    predictor.seen=0
    predictor.args.plots=False
    predictor.stats={}
    predictor.stats['tp']=[]
    pbatch = predictor._prepare_batch(0, batch)
    cls, bbox = pbatch.pop("cls"), pbatch.pop("bbox")
    predn = predictor._prepare_pred(pred, pbatch)
    iou_mat = box_iou(bbox, predn[:, :4])
    if iou_mat.numel() == 0 or iou_mat.shape[1] == 0 or iou_mat.shape[0] == 0:
        return np.zeros(1)

    mean_iou_per_image =   (iou_mat*(iou_mat==iou_mat.max(axis=0, keepdim=True).values)).max(axis=1).values.numpy()

    return np.expand_dims(mean_iou_per_image.mean(),axis=0)


@tensorleap_custom_instances_metric("example_instance_metric", direction=MetricDirection.Upward)
def example_custom_instance_metric(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    sample_id = preprocess.sample_ids[0]
    n_instances = instances_length_encoder(str(sample_id), preprocess.preprocess_response)
    return {i: np.random.rand(1).astype(np.float32) for i in range(n_instances)}


@tensorleap_custom_instances_metric("instance_best_iou", direction=MetricDirection.Upward)
def instance_best_iou(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    """Per-GT-instance IoU of the best-matching prediction (0 if unmatched)."""
    sample_id = preprocess.sample_ids[0]
    n_instances = instances_length_encoder(str(sample_id), preprocess.preprocess_response)
    result = {i: np.zeros(1, dtype=np.float32) for i in range(n_instances)}
    if n_instances == 0:
        return result

    batch = preprocess.preprocess_response.data['dataloader'][int(sample_id)]
    batch["imgsz"] = (batch["resized_shape"],)
    batch["ori_shape"] = (batch["ori_shape"],)
    batch["ratio_pad"] = (batch["ratio_pad"],)
    batch["img"] = batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred))[0]
    predictor.seen = 0
    predictor.args.plots = False
    predictor.stats = {'tp': []}
    pbatch = predictor._prepare_batch(0, batch)
    gt_cls, gt_bbox = pbatch.pop("cls"), pbatch.pop("bbox")
    predn = predictor._prepare_pred(pred, pbatch)

    if predn.shape[0] == 0 or gt_bbox.shape[0] == 0:
        return result

    iou_mat = box_iou(gt_bbox, predn[:, :4])
    # Restrict matches to predictions whose class equals the GT class.
    same_class = (gt_cls.view(-1, 1) == predn[:, 5].view(1, -1))
    iou_mat = iou_mat * same_class.to(iou_mat.dtype)

    best_iou_per_gt = iou_mat.max(dim=1).values.numpy()
    for i in range(min(n_instances, best_iou_per_gt.shape[0])):
        result[i] = np.array([best_iou_per_gt[i]], dtype=np.float32)
    return result


@tensorleap_custom_instances_metric("instance_match_confidence", direction=MetricDirection.Upward)
def instance_match_confidence(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
    """Per-GT-instance confidence of the best-matching same-class prediction (0 if unmatched)."""
    sample_id = preprocess.sample_ids[0]
    n_instances = instances_length_encoder(str(sample_id), preprocess.preprocess_response)
    result = {i: np.zeros(1, dtype=np.float32) for i in range(n_instances)}
    if n_instances == 0:
        return result

    batch = preprocess.preprocess_response.data['dataloader'][int(sample_id)]
    batch["imgsz"] = (batch["resized_shape"],)
    batch["ori_shape"] = (batch["ori_shape"],)
    batch["ratio_pad"] = (batch["ratio_pad"],)
    batch["img"] = batch["img"].unsqueeze(0)
    pred = predictor.postprocess(torch.from_numpy(y_pred))[0]
    predictor.seen = 0
    predictor.args.plots = False
    predictor.stats = {'tp': []}
    pbatch = predictor._prepare_batch(0, batch)
    gt_cls, gt_bbox = pbatch.pop("cls"), pbatch.pop("bbox")
    predn = predictor._prepare_pred(pred, pbatch)

    if predn.shape[0] == 0 or gt_bbox.shape[0] == 0:
        return result

    iou_mat = box_iou(gt_bbox, predn[:, :4])
    same_class = (gt_cls.view(-1, 1) == predn[:, 5].view(1, -1))
    iou_mat = iou_mat * same_class.to(iou_mat.dtype)

    best_pred_idx = iou_mat.argmax(dim=1).numpy()
    best_iou_per_gt = iou_mat.max(dim=1).values.numpy()
    confidences = predn[:, 4].numpy()
    for i in range(min(n_instances, best_pred_idx.shape[0])):
        if best_iou_per_gt[i] > 0:
            result[i] = np.array([confidences[best_pred_idx[i]]], dtype=np.float32)
    return result





# @tensorleap_custom_metric("cost", direction=MetricDirection.Downward)
# def cost(pred80,pred40,pred20,gt):
#     gt=np.squeeze(gt,axis=0)
#     d={}
#     d["bboxes"] = torch.from_numpy(gt[...,:4])
#     d["cls"] = torch.from_numpy(gt[...,4])
#     d["batch_idx"] = torch.zeros_like(d['cls'])
#     y_pred_torch = [torch.from_numpy(s) for s in [pred80,pred40,pred20]]
#     _,loss_parts= criterion(y_pred_torch, d)
#     return {"box":loss_parts[0].unsqueeze(0).numpy(),"cls":loss_parts[1].unsqueeze(0).numpy(),"dfl":loss_parts[2].unsqueeze(0).numpy()}


#
# leap_binder.add_prediction(name='object detection', labels=["x", "y", "w", "h"] + [cl for cl in all_clss.values()], channel_dim=1)
# leap_binder.add_prediction(name='concatenate_20', labels=[str(i) for i in range(20)], channel_dim=-1)
# leap_binder.add_prediction(name='concatenate_40', labels=[str(i) for i in range(40)], channel_dim=-1)
# leap_binder.add_prediction(name='concatenate_80', labels=[str(i) for i in range(80)], channel_dim=-1)

if __name__ == '__main__':
    leap_binder.check()

