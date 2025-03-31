
from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         leap_binder, loss, metadata_sample_index, gt_bb_decoder, image_visualizer, bb_decoder)
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
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)
            y_pred = model([concat])
            gt = gt_encoder(idx, subset)
            loss_array=loss(y_pred,gt) #TODO - fix in tensorleap (check if list is acceptable)

            img_vis=image_visualizer(image)
            gt_img=gt_bb_decoder(image,gt)
            # pred_img=bb_decoder(image,y_pred[0].numpy().squeeze()) # TODO - fix in tensorleap

            if plot_vis:
                visualize(img_vis)
                visualize(gt_img)
                # visualize(pred_img)

            metadata_sample=metadata_sample_index(idx,subset)
    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
