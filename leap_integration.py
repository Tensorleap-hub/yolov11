import os
from code_loader.contract.datasetclasses import PredictionTypeHandler
from code_loader.inner_leap_binder.leapbinder_decorators import tensorleap_load_model, tensorleap_integration_test
from code_loader.plot_functions.visualize import visualize

from leap_binder import input_encoder, image_visualizer, bb_decoder, gt_encoder, gt_bb_decoder, preprocess_func_leap

import tensorflow as tf

from ultralytics.tensorleap_folder.global_params import all_clss

prediction_type1 = PredictionTypeHandler('object detection', labels=["x", "y", "w", "h"] + [cl for cl in all_clss.values()], channel_dim=1)
prediction_type2 = PredictionTypeHandler('concatenate_80', labels=[str(i) for i in range(80)], channel_dim=-1)
prediction_type3 = PredictionTypeHandler('concatenate_80', labels=[str(i) for i in range(40)], channel_dim=-1)
prediction_type4 = PredictionTypeHandler('concatenate_80', labels=[str(i) for i in range(20)], channel_dim=-1)





@tensorleap_load_model([prediction_type1, prediction_type2, prediction_type3, prediction_type4])
def load_model():
    dir_path = os.path.dirname(os.path.abspath(__file__))
    model_path = './yolov11sb.h5'
    cnn = tf.keras.models.load_model(os.path.join(dir_path, model_path))
    return cnn


@tensorleap_integration_test()
def check_custom_test_mapping(idx, subset):
    model = load_model()

    image = input_encoder(idx, subset)

    y_pred = model([image])
    img_vis = image_visualizer(image)
    pred_img = bb_decoder(image, y_pred[0])

    gt = gt_encoder(idx, subset)

    gt_img = gt_bb_decoder(image, gt)

    visualize(img_vis)
    visualize(pred_img)
    visualize(gt_img)






if __name__ == '__main__':
    preprocess_resoinse = preprocess_func_leap()[0]
    sample_id = preprocess_resoinse.sample_ids[0]
    check_custom_test_mapping(sample_id, preprocess_func_leap()[0])












