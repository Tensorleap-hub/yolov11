import torch
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_custom_loss, tensorleap_custom_metric
from ultralytics.tensorleap_folder.global_params import cfg, yolo_data, criterion, all_clss, predictor
from ultralytics.tensorleap_folder.utils import create_data_with_ult, pre_process_dataloader, update_dict_bbox_cls_info, \
    update_dict_count_cls, bbox_area_and_aspect_ratio, calculate_iou_all_pairs
from typing import List, Dict, Union
import numpy as np
from code_loader import leap_binder
from code_loader.contract.datasetclasses import PreprocessResponse, DataStateType, SamplePreprocessResponse
from code_loader.contract.enums import LeapDataType, MetricDirection
from code_loader.visualizers.default_visualizers import LeapImage
from code_loader.inner_leap_binder.leapbinder_decorators import (tensorleap_preprocess, tensorleap_gt_encoder,
                                                                 tensorleap_input_encoder, tensorleap_metadata,
                                                                 tensorleap_custom_visualizer)
from code_loader.contract.responsedataclasses import BoundingBox
from code_loader.contract.visualizer_classes import LeapImageWithBBox
from code_loader.utils import rescale_min_max
from ultralytics.utils.plotting import output_to_target
from ultralytics.utils.metrics import box_iou



# ----------------------------------------------------data processing---------------------------------------------------

@tensorleap_preprocess()
def preprocess_func_leap() -> List[PreprocessResponse]:
    data_loader_val,n_sampels_val=create_data_with_ult(cfg,yolo_data,phase='val')
    data_loader_train,n_sampels_train=create_data_with_ult(cfg,yolo_data,phase='train')

    val = PreprocessResponse(sample_ids=list(range(n_sampels_val))[:1000], data={'dataloader':data_loader_val},sample_id_type=int, state=DataStateType.validation)
    train = PreprocessResponse(sample_ids=list(range(n_sampels_train))[:1000], data={'dataloader':data_loader_train},sample_id_type=int, state=DataStateType.training)
    response = [val,train]
    return response


# ------------------------------------------input and gt----------------------------------------------------------------


# Input encoder fetches the image with the index `idx` from the `images` array set in
# the PreprocessResponse data. Returns a numpy array containing the sample's image.
@tensorleap_input_encoder('image',channel_dim=1)
def input_encoder(idx: int, preprocess: PreprocessResponse) -> np.ndarray:
    imgs, _, _,_=pre_process_dataloader(preprocess, idx, predictor)

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
    _, clss, bboxes, _ =pre_process_dataloader(preprocessing, idx,predictor)
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
def metadata_sample_index(idx: int, preprocess: PreprocessResponse) -> int:
    return idx


@tensorleap_metadata("image info a")
def metadata_per_img(idx: int, data: PreprocessResponse) -> Dict[str, Union[str, int, float]]:
    nan_default_value=-1.
    gt_data=gt_encoder(idx, data)
    cls_gt=np.expand_dims(gt_data[:,4],axis=1)
    bbox_gt=gt_data[:,:4]
    clss_info=np.unique(cls_gt,return_counts=True)
    count_dict=update_dict_count_cls(all_clss, clss_info)
    areas, aspect_ratios=bbox_area_and_aspect_ratio(bbox_gt,data.data['dataloader'][idx]['resized_shape'])
    occlusion_matrix, areas_in_pixels, union_in_pixels=calculate_iou_all_pairs(bbox_gt, data.data['dataloader'][idx]['resized_shape'])
    non_zeros_occlusion_mask= occlusion_matrix.sum(axis=1) != 0
    num_pix_in_im=(data.data['dataloader'][idx]['resized_shape'][0]*data.data['dataloader'][idx]['resized_shape'][1])
    no_nans_values= ~np.isnan(clss_info[0]).any()
    x_center, y_center = bbox_gt[:, 0], bbox_gt[:, 1]

    d = {
            "image path": data.data['dataloader'].im_files[idx],
            "idx":idx,
            "# unique classes" : len(clss_info[0]) if no_nans_values else nan_default_value,
            "# of objects": int(clss_info[1].sum()) if no_nans_values else nan_default_value,
            "mean bbox x loc": float(x_center.mean()) if no_nans_values else nan_default_value,
            "var bbox x loc": float(x_center.var()) if no_nans_values else nan_default_value,
            "mean bbox y loc": float(y_center.mean()) if no_nans_values else nan_default_value,
            "var bbox y loc": float(y_center.var()) if no_nans_values else nan_default_value,
            "mean bbox area": float(areas.mean()) if no_nans_values else nan_default_value,
            "var bbox area": float(areas.var()) if no_nans_values else nan_default_value,
            "mean aspect ratio": float(aspect_ratios.mean()) if no_nans_values else nan_default_value,
            "var aspect ratio": float(aspect_ratios.var()) if no_nans_values else nan_default_value,
            "bbox occlusion": float(occlusion_matrix.sum()/num_pix_in_im) if no_nans_values else nan_default_value,
            "var bbox occlusion": float(occlusion_matrix.sum(axis=1)[non_zeros_occlusion_mask].var() if np.sum(non_zeros_occlusion_mask)>0 else 0. /num_pix_in_im) if no_nans_values else nan_default_value,
            "max bbox occlusion": float(occlusion_matrix.sum(axis=1).max()/num_pix_in_im) if no_nans_values else nan_default_value,
            "bbox occlusion/union": float(occlusion_matrix.sum() / areas_in_pixels.sum()) if no_nans_values else nan_default_value,
            "max bbox occlusion/union": float((occlusion_matrix.sum(axis=1)/ areas_in_pixels).max()) if no_nans_values else nan_default_value,

        }

    # feature_map1 = {'area': areas, 'ar': aspect_ratios,'occlusion': occlusion_matrix/num_pix_in_im}
    # func_types1 = ['mean', 'max', 'min']
    #
    # for feat_name, feat_data in feature_map1.items():
    #     for func_type in func_types1:
    #         result_dict = update_dict_bbox_cls_info(all_clss,feat_data,cls_gt,func_type,feat_name,nan_default_value)
    #         d.update(**result_dict)

    # feature_map2 = {'x_center': x_center, 'y_center': y_center}
    #     func_types2 = ['mean', 'var', 'max', 'min']
    #
    #     for feat_name, feat_data in feature_map2.items():
    #         for func_type in func_types2:
    #             result_dict = update_dict_bbox_cls_info(all_clss,feat_data,cls_gt,func_type,feat_name,nan_default_value)
    #             d.update(**result_dict)

    d.update(**count_dict)
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

@tensorleap_custom_metric("cost", direction=MetricDirection.Downward)
def cost(pred80,pred40,pred20,gt):
    gt=np.squeeze(gt,axis=0)
    d={}
    d["bboxes"] = torch.from_numpy(gt[...,:4])
    d["cls"] = torch.from_numpy(gt[...,4])
    d["batch_idx"] = torch.zeros_like(d['cls'])
    y_pred_torch = [torch.from_numpy(s) for s in [pred80,pred40,pred20]]
    _,loss_parts= criterion(y_pred_torch, d)
    return {"box":loss_parts[0].unsqueeze(0).numpy(),"cls":loss_parts[1].unsqueeze(0).numpy(),"dfl":loss_parts[2].unsqueeze(0).numpy()}


# @tensorleap_custom_metric("metadata_metric",compute_insights=False)
# def metadata_metric(y_pred: np.ndarray, preprocess: SamplePreprocessResponse):
#     nan_default_value=-1.
#     preds=predictor.postprocess(torch.from_numpy(y_pred))[0]
#     idx=int(preprocess.sample_ids)
#     cls = np.expand_dims(preds[:, 5], axis=1)
#     bbox = preds[:, :4].numpy()
#     clss_info = np.unique(cls, return_counts=True)
#     count_dict = update_dict_count_cls(all_clss, clss_info)
#     areas, aspect_ratios = bbox_area_and_aspect_ratio(bbox, preprocess.preprocess_response.data['dataloader'][idx]['resized_shape'])
#     occlusion_matrix, areas_in_pixels, union_in_pixels = calculate_iou_all_pairs(bbox, preprocess.preprocess_response.data['dataloader'][idx][
#         'resized_shape'])
#     non_zeros_occlusion_mask = occlusion_matrix.sum(axis=1) != 0
#     num_pix_in_im = (
#                 preprocess.preprocess_response.data['dataloader'][idx]['resized_shape'][0] * preprocess.preprocess_response.data['dataloader'][idx]['resized_shape'][1])
#     no_nans_values = ~np.isnan(clss_info[0]).any()
#     x_center, y_center = bbox[:, 0], bbox[:, 1]
#
#     d = {
#         "# unique classes": len(clss_info[0]) if no_nans_values else nan_default_value,
#         "# of objects": int(clss_info[1].sum()) if no_nans_values else nan_default_value,
#         "mean bbox x loc": float(x_center.mean()) if no_nans_values else nan_default_value,
#         "var bbox x loc": float(x_center.var()) if no_nans_values else nan_default_value,
#         "mean bbox y loc": float(y_center.mean()) if no_nans_values else nan_default_value,
#         "var bbox y loc": float(y_center.var()) if no_nans_values else nan_default_value,
#         "mean bbox area": float(areas.mean()) if no_nans_values else nan_default_value,
#         "var bbox area": float(areas.var()) if no_nans_values else nan_default_value,
#         "mean aspect ratio": float(aspect_ratios.mean()) if no_nans_values else nan_default_value,
#         "var aspect ratio": float(aspect_ratios.var()) if no_nans_values else nan_default_value,
#         "bbox occlusion": float(occlusion_matrix.sum() / num_pix_in_im) if no_nans_values else nan_default_value,
#         "var bbox occlusion": float(occlusion_matrix.sum(axis=1)[non_zeros_occlusion_mask].var() if np.sum(
#             non_zeros_occlusion_mask) > 0 else 0. / num_pix_in_im) if no_nans_values else nan_default_value,
#         "max bbox occlusion": float(
#             occlusion_matrix.sum(axis=1).max() / num_pix_in_im) if no_nans_values else nan_default_value,
#         "bbox occlusion/union": float(
#             occlusion_matrix.sum() / areas_in_pixels.sum()) if no_nans_values else nan_default_value,
#         "max bbox occlusion/union": float(
#             (occlusion_matrix.sum(axis=1) / areas_in_pixels).max()) if no_nans_values else nan_default_value,
#
#     }
#     d.update(**count_dict)
#     for key in d:
#         d[key] = np.array([d[key]])
#     return d


# ---------------------------------------------------------main------------------------------------------------------



leap_binder.add_prediction(name='object detection', labels=["x", "y", "w", "h"] + [cl for cl in all_clss.values()], channel_dim=1)
leap_binder.add_prediction(name='concatenate_20', labels=[str(i) for i in range(20)], channel_dim=-1)
leap_binder.add_prediction(name='concatenate_40', labels=[str(i) for i in range(40)], channel_dim=-1)
leap_binder.add_prediction(name='concatenate_80', labels=[str(i) for i in range(80)], channel_dim=-1)

if __name__ == '__main__':
    leap_binder.check()

