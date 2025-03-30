# import torch

from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         leap_binder, dummy_loss, metadata_sample_index, gt_bb_decoder, image_visualizer, bb_decoder)
import tensorflow as tf
import numpy as np
from code_loader.helpers import visualize

# from ultralytics.utils import ops
# from ultralytics.utils.plotting import output_to_target


def check_custom_test():
    check_generic = True
    plot_vis= True
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    model_path = r"yolov11s.h5"
    model = tf.keras.models.load_model(model_path)

    responses = preprocess_func_leap()
    for subset in responses:
        # detector= subset.data['predictor']
        for idx in range(10):
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)
            y_pred = model([concat]).numpy()
            gt = gt_encoder(idx, subset)


            # y_pred= detector.postprocess(torch.from_numpy(y_pred.numpy()))
            # _,cls_temp,bbx_temp,conf_temp=output_to_target(y_pred, max_det=detector.args.max_det)
            # t_pred=np.concatenate([bbx_temp,np.expand_dims(conf_temp,1),np.expand_dims(cls_temp,1)],axis=1)
            # post_proc_pred = t_pred[t_pred[:, 4] > 0.25] # TODO- make this a cfg param
            # post_proc_pred[:,:4:2]/=image.shape[1]
            # post_proc_pred[:,1:4:2]/=image.shape[2]


            img_vis=image_visualizer(image)
            gt_img=gt_bb_decoder(image,gt)
            pred_img=bb_decoder(image,y_pred.squeeze())

            if plot_vis:
                visualize(img_vis)
                visualize(gt_img)
                visualize(pred_img)


            d_loss=dummy_loss(y_pred,gt)
            metadata_sample=metadata_sample_index(idx,subset)
    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
