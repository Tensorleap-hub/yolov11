from typing import List, Dict
from collections import defaultdict
from super_gradients.training.metrics import DetectionMetrics_050
import torch

from code_loader.contract.datasetclasses import ConfusionMatrixElement
from code_loader.contract.enums import ConfusionMatrixValue
from code_loader.inner_leap_binder.leapbinder_decorators import *

from tensorleap.config import CONFIG
from tensorleap.utils import prep_sg_inputs, nms

# here we map between the ids of labels to their visualized name
id_to_name = CONFIG['classid2name']


def _elements_for_single_image(
        pred_boxes: torch.Tensor,  # (N_pred, 4)
        pred_scores: torch.Tensor,  # (N_pred, C)
        targets: torch.Tensor,  # (N_gt, 6)  id,label,cx,cy,w,h
        id_to_name: Dict[int, str],
        metric: DetectionMetrics_050,
        background_label: str = "background"
) -> List[ConfusionMatrixElement]:

    metric.reset()
    metric.update(preds=((pred_boxes[None, ...], pred_scores[None, ...]), 0),
                  target=targets,
                  inputs=torch.randn(1, 3, CONFIG['image_h'], CONFIG['image_w']),
                  device=str(metric.device))

    cm_elems: List[ConfusionMatrixElement] = []
    # read the private DetectionMetrics_050 cache
    for state_key in getattr(metric, metric.state_key):
        # read the private cache  -----------------------------------------------
        preds_matched, preds_to_ignore, scores, preds_cls, targets_cls = state_key
        matched_pred_labels = defaultdict(int)
        gt_label_count = defaultdict(int)
        for i in targets_cls:
            gt_label_count[int(i)] += 1
        for conf, is_tp, is_ign, cls_id in zip(scores, preds_matched, preds_to_ignore, preds_cls):

            # TP & FP
            if is_ign:
                continue
            outcome = (ConfusionMatrixValue.Positive if is_tp
                       else ConfusionMatrixValue.Negative)
            label = id_to_name[int(cls_id)]
            cm_elems.append(ConfusionMatrixElement(label, outcome, float(conf)))
            if is_tp:
                matched_pred_labels[int(cls_id)] += 1

        # FN count no prediction
        for lbl_id in id_to_name.keys():
            tp_cnt = matched_pred_labels.get(lbl_id, 0)
            gt_cnt = gt_label_count.get(lbl_id, 0)
            fn_cnt = gt_cnt - tp_cnt
            cm_elems.extend(ConfusionMatrixElement(id_to_name[lbl_id],
                                                   ConfusionMatrixValue.Positive,
                                                   0.0)
                            for _ in range(fn_cnt))

        # fallback when cache is empty or all elems were ignored
        if not cm_elems:
            cm_elems.append(
                ConfusionMatrixElement(background_label,
                                       ConfusionMatrixValue.Positive,
                                       0.0)
            )

    metric.reset()  # ready for next image
    return cm_elems


@tensorleap_custom_metric('Confusion Matrix')
def confusion_matrix_metric(pred_boxes: np.ndarray, pred_scores: np.ndarray, targets: np.ndarray):
    device = "cpu"
    metric = DetectionMetrics_050(
        score_thres=0.5,  # Only predictions with confidence >= 0.5 will be considered.
        top_k_predictions=CONFIG['nms']['nms_top_k'],  # Up to 128 predictions per image are evaluated.
        num_cls=CONFIG['num_classes'],  # Suppose we have 3 classes.
        normalize_targets=True,  # Assumes target boxes are normalized.
        post_prediction_callback=nms
    )

    _, targets_batch = prep_sg_inputs(pred_boxes, pred_scores, targets)
    pred_boxes_batch = torch.from_numpy(pred_boxes)   # (B, N, 4)
    pred_scores_batch = torch.from_numpy(pred_scores)  # (B, N, C)
    targets_batch = targets_batch[None, ...]

    all_results = []
    for preds_box_i, preds_score_i,  targets_i in zip(pred_boxes_batch, pred_scores_batch, targets_batch):
        all_results.append(
            _elements_for_single_image(preds_box_i.to(device),
                                       preds_score_i.to(device),
                                       targets_i.to(device),
                                       id_to_name,
                                       metric)
        )
    return all_results