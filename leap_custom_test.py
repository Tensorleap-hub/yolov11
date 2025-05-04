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
    if check_generic:
        leap_binder.check()
    m_path= model_path if model_path!=None else 'None_path'
    print("started custom tests")
    if not os.path.exists(m_path):
        from export_model_to_tf import start_export #TODO - currently supports only onnx
        m_path=start_export()
    keras_model=m_path.endswith(".h5")
    model = tf.keras.models.load_model(m_path) if keras_model else ort.InferenceSession(m_path)

    responses = preprocess_func_leap()
    for subset in responses:
        for idx in range(20):
            s_prepro=SamplePreprocessResponse(np.array(idx), subset)
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)
            meta_data=misc_metadata(idx, subset)
            y_pred = model([concat]) if keras_model else model.run(None, {model.get_inputs()[0].name: concat})
            if not keras_model:
                y_pred=[tf.convert_to_tensor(p)  for p in y_pred]
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
    check_generic = True
    plot_vis= True
    model_path = r'/Users/yamtawachi/tensorleap/datasets/models/yolo11sb.h5' # Choose None if only pt version available or your  h5/onnx model's path.


    check_custom_test()
