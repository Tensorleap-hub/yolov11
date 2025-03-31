# import torch
import torch
from charset_normalizer import detect

from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         leap_binder, dummy_loss, metadata_sample_index, gt_bb_decoder, image_visualizer, bb_decoder)
import tensorflow as tf
import numpy as np
from code_loader.helpers import visualize
from code_loader.helpers.detection.yolo.utils import reshape_output_list
from ultralytics.tensorleap_folder.config import cfg
from code_loader.helpers.detection.yolo.enums import YoloDecodingType



def check_custom_test():
    check_generic = False
    plot_vis= True
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    model_path = r"yolov11sb.h5"
    model = tf.keras.models.load_model(model_path)


    # model_base.args=cfg

    responses = preprocess_func_leap()
    for subset in responses:
        for idx in range(10):
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)
            y_pred = model([concat])
            gt = gt_encoder(idx, subset)
            loss_array=dummy_loss(y_pred,gt)
            img_vis=image_visualizer(image)
            gt_img=gt_bb_decoder(image,gt)
            # pred_img=bb_decoder(image,y_pred)

            # if plot_vis:
                # visualize(img_vis)
                # visualize(gt_img)
                # visualize(pred_img)


            metadata_sample=metadata_sample_index(idx,subset)
    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
