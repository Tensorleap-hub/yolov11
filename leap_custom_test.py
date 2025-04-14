from code_loader.contract.datasetclasses import SamplePreprocessResponse

from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         leap_binder, loss, gt_bb_decoder, image_visualizer, bb_decoder,
                         iou_dic, cost, metadata_metric, metadata_per_img)
import tensorflow as tf
import numpy as np
from code_loader.helpers import visualize



def check_custom_test():
    check_generic = True
    plot_vis= True
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    model_path = r"yolov11sb.h5"
    model = tf.keras.models.load_model(model_path)

    responses = preprocess_func_leap()
    for subset in responses:
        for idx in range(10):
            s_prepro=SamplePreprocessResponse(np.array(idx), subset)
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)
            gt = gt_encoder(idx, subset)
            meta_data1=metadata_per_img(idx, subset)

            y_pred = model([concat])

            d_metrics=metadata_metric(y_pred[0].numpy(),s_prepro)

            iou=iou_dic(y_pred[0].numpy(), s_prepro)
            total_loss=loss(y_pred[1].numpy(),y_pred[2].numpy(),y_pred[3].numpy(),np.expand_dims(gt,axis=0), y_pred[0].numpy())
            cost_dic=cost(y_pred[1].numpy(),y_pred[2].numpy(),y_pred[3].numpy(),np.expand_dims(gt,axis=0))
            img_vis=image_visualizer(np.expand_dims(image,axis=0))
            gt_img=gt_bb_decoder(np.expand_dims(image,axis=0),np.expand_dims(gt,axis=0))
            pred_img=bb_decoder(np.expand_dims(image,axis=0),y_pred[0].numpy())
            if plot_vis:
                visualize(img_vis)
                visualize(gt_img)
                visualize(pred_img)
    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
