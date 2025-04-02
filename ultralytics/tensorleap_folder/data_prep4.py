from coremltools.converters.mil.testing_reqs import tf

from ultralytics.data import build_yolo_dataset
from ultralytics.models.yolo.model import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.build import build_dataloader
from ultralytics.utils import callbacks
from ultralytics.models.yolo.detect.predict import DetectionPredictor
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import Annotator, colors
import torch
import cv2
from global_params import cfg
from ultralytics.utils import ASSETS
from ultralytics.models.yolo.detect import DetectionPredictor
from ultralytics.utils.torch_utils import de_parallel
from ultralytics.utils.plotting import plot_images, output_to_target
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

import matplotlib.pyplot as plt




# Plot the image

dataset_path="coco.yaml"

callbacks = callbacks.get_default_callbacks()
data =check_det_dataset(dataset_path, autodownload=True)
dataset=build_yolo_dataset(cfg, data['val'], 1, data, mode='val', stride=32)
dataloader=build_dataloader(dataset, 1, 0, shuffle=False, rank=-1)# build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)


### predictor object- is it ok?
predictor = DetectionValidator(args=cfg, _callbacks=callbacks)
predictor.data=data
predictor.dataloader=dataloader

# predictor = DetectionPredictor(args=cfg)

# args = dict(model="yolo11n.pt", source=ASSETS)
predictor_pred = DetectionPredictor(overrides=cfg)

# 4. Load model
model_pt = YOLO("yolo11n.pt").model
model_path = r"/Users/yamtawachi/tensorleap/ultralytics/yolov11s.h5"
model_tf = tf.keras.models.load_model(model_path)
predictor.init_metrics(de_parallel(model_pt))

# 5. Start evaluation loop
for batch_i, batch in enumerate(dataloader):

    predictor.run_callbacks("on_val_batch_start")
    batch = predictor.preprocess(batch)

    preds_pt = model_pt(batch["img"])

    preds_tf = model_tf(batch["img"])
    preds_tf=torch.from_numpy(preds_tf.numpy())
    import numpy as np

    print(np.abs(preds_tf - preds_pt[0]).mean())

    preds_pt=predictor.postprocess(preds_pt)
    preds_tf  = predictor.postprocess(preds_tf )

    predictor.update_metrics(preds_pt, batch)
    predictor.update_metrics(preds_tf , batch)

    # pred_samp_plot_pt=plot_images(
    #     batch["img"],
    #     *output_to_target(preds_pt, max_det=predictor.args.max_det),
    #     paths=batch["im_file"],
    #     fname=predictor.save_dir / f"val_batch{batch_i}_pred.jpg",
    #     names=predictor.names,
    #     on_plot=predictor.on_plot,
    #     save=False,
    #     threaded=False
    # )
    # plt.imshow(pred_samp_plot_pt)
    # plt.axis("off")
    # plt.savefig(f"sample_{batch_i}_pred_pt.png")
    #
    # val_smps_plot_pt=plot_images(batch["img"],batch["batch_idx"], batch["cls"].squeeze(-1),batch["bboxes"], names=predictor.names,save=False,threaded=False)
    # plt.imshow(val_smps_plot_pt)
    # plt.axis("off")
    # plt.savefig(f"sample_{batch_i}_gt_pt.png")

    pred_samp_plot_tf  = plot_images(
        batch["img"],
        *output_to_target(preds_tf , max_det=predictor.args.max_det),
        paths=batch["im_file"],
        fname=predictor.save_dir / f"val_batch{batch_i}_pred.jpg",
        names=predictor.names,
        on_plot=predictor.on_plot,
        save=False,
        threaded=False
    )
    plt.imshow(pred_samp_plot_tf )
    plt.axis("off")
    plt.savefig(f"sample_{batch_i}_pred_tf .png")

    val_smps_plot_tf  = plot_images(batch["img"], batch["batch_idx"], batch["cls"].squeeze(-1), batch["bboxes"],
                                   paths=batch["im_file"], fname=predictor.save_dir / f"val_batch{batch_i}_labels.jpg",
                                   names=predictor.names, on_plot=predictor.on_plot, save=False, threaded=False)
    plt.imshow(val_smps_plot_tf )
    plt.axis("off")
    plt.savefig(f"sample_{batch_i}_gt_tf.png")






