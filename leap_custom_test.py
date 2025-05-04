import os

from code_loader.contract.datasetclasses import SamplePreprocessResponse
from code_loader.contract.enums import DataStateType

from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         leap_binder, loss, gt_bb_decoder, image_visualizer, bb_decoder,
                         misc_metadata, iou_dic, cost)
import tensorflow as tf
import onnxruntime as ort
import numpy as np
from code_loader.helpers import visualize



def check_custom_test():
    check_generic = True
    plot_vis= True
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    model_path = r"yolov11m.onnx"
    if not os.path.exists(model_path):
        from export_model_to_tf import start_export #TODO - currently supports only onnx
        model_path=start_export

    model = tf.keras.models.load_model(model_path) if model_path.endswith(".h5") else ort.InferenceSession(model_path)

    responses = preprocess_func_leap()
    for subset in responses:
        for idx in range(2):
            s_prepro=SamplePreprocessResponse(np.array(idx), subset)
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)
            meta_data=misc_metadata(idx, subset)
            y_pred = model([concat])
            if subset.state != DataStateType.unlabeled:
                iou = iou_dic(y_pred[0].numpy(), s_prepro)
                gt = gt_encoder(idx, subset)
                total_loss=loss(y_pred[1].numpy(),y_pred[2].numpy(),y_pred[3].numpy(),np.expand_dims(gt,axis=0), y_pred[0].numpy())
                cost_dic=cost(y_pred[1].numpy(),y_pred[2].numpy(),y_pred[3].numpy(),np.expand_dims(gt,axis=0))
                gt_img = gt_bb_decoder(np.expand_dims(image, axis=0), np.expand_dims(gt, axis=0))
            img_vis=image_visualizer(np.expand_dims(image,axis=0))
            pred_img=bb_decoder(np.expand_dims(image,axis=0),y_pred[0].numpy())
            if plot_vis:
                visualize(img_vis)
                visualize(pred_img)
                if subset.state != DataStateType.unlabeled:
                    visualize(gt_img)

    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
