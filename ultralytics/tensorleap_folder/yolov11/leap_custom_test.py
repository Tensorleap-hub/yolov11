from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         leap_binder,dummy_loss,metadata_sample_index)
import tensorflow as tf
import os
import numpy as np


def check_custom_test():
    check_generic = False
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    model_path = r"yolov11s.h5"
    cnn = tf.keras.models.load_model(model_path)

    responses = preprocess_func_leap()
    for subset in responses:
        for idx in range(10):
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)
            y_pred = cnn([concat])
            gt = gt_encoder(idx, subset)
            d_loss=dummy_loss(y_pred.numpy(),gt)
            metadata_sample=metadata_sample_index(idx,subset)
    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
