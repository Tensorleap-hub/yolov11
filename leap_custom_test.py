from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         leap_binder,dummy_loss,metadata_sample_index,gt_bb_decoder,image_visualizer)
import tensorflow as tf
import numpy as np
from code_loader.helpers import visualize


def check_custom_test():
    check_generic = False
    plot_vis= True
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    model_path = r"yolov11s.h5"
    model = tf.keras.models.load_model(model_path)

    responses = preprocess_func_leap()
    for subset in responses:
        for idx in range(10):
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)
            y_pred = model([concat])
            gt = gt_encoder(idx, subset)

            img_vis=image_visualizer(image)
            # gt_img=gt_bb_decoder(image,gt[:,:-1])

            if plot_vis:
                visualize(img_vis)

            d_loss=dummy_loss(y_pred.numpy(),gt)
            metadata_sample=metadata_sample_index(idx,subset)
    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
