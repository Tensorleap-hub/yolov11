from coremltools.converters.mil.testing_reqs import tf
from matplotlib import pyplot as plt
from ultralytics.data import build_dataloader, build_yolo_dataset, converter
from ultralytics.models.yolo.model import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.converter import convert_coco
from ultralytics.data.build import build_dataloader
from ultralytics.utils import callbacks
from ultralytics.models.yolo.detect.val import DetectionValidator
from ultralytics.utils.torch_utils import de_parallel
from config import cfg
import copy


dataset_path="coco.yaml"

callbacks = callbacks.get_default_callbacks()
data =check_det_dataset(dataset_path, autodownload=True)
dataset=build_yolo_dataset(cfg, data['val'], 1, data, mode='val', stride=32)
dataloader=build_dataloader(dataset, len(dataset), 0, shuffle=False, rank=-1)# build_yolo_dataset(self.args, img_path, batch, self.data, mode=mode, stride=self.stride)


### predictor object- is it ok?
predictor = DetectionValidator(args=cfg, _callbacks=callbacks)
predictor.data=data
predictor.dataloader=dataloader


##### model demo ##
model_pt = YOLO("tensorleap_folder/yolo11n.pt").model
model_path = r"/Users/yamtawachi/tensorleap/ultralytics/yolov11/yolov11s.h5"
model_tf = tf.keras.models.load_model(model_path)

batch = next(iter(dataloader))

 ## same pipline as in ultralytics/engine/validator.py   basevalidator __call__
for batch_i, batch in enumerate(dataloader):
    preprocessed_batch=predictor.preprocess(batch)
    preds_tf = model_tf(batch["img"]).numpy()
    preds_pt=model_pt(batch["img"])

    predictor.init_metrics(de_parallel(model_pt))
    # model.loss(batch, preds)[1] # ????
    # predictor.postprocess(preds) # nms
    predictor.update_metrics(preds, batch)
    if predictor.args.plots and batch_i < 3:
        predictor.plot_val_samples(batch, batch_i)
        predictor.plot_predictions(batch, preds, predictor)
    a=5



#
#
#
# callbacks("on_val_start")
#         dt = (
#             Profile(device=self.device),
#             Profile(device=self.device),
#             Profile(device=self.device),
#             Profile(device=self.device),
#         )
#         bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
#         self.init_metrics(de_parallel(model))
#         self.jdict = []  # empty before each val
#         for batch_i, batch in enumerate(bar):
#             self.run_callbacks("on_val_batch_start")
#             self.batch_i = batch_i
#             # Preprocess
#             with dt[0]:
#                 batch = self.preprocess(batch)
#
#             # Inference
#             with dt[1]:
#                 preds = model(batch["img"], augment=augment)
#
#             # Loss
#             with dt[2]:
#                 if self.training:
#                     self.loss += model.loss(batch, preds)[1]
#
#             # Postprocess
#             with dt[3]:
#                 preds = self.postprocess(preds)
#
#             self.update_metrics(preds, batch)
#             if self.args.plots and batch_i < 3:
#                 self.plot_val_samples(batch, batch_i)
#                 self.plot_predictions(batch, preds, batch_i)
#
#             self.run_callbacks("on_val_batch_end")
#         stats = self.get_stats()
#         self.check_stats(stats)
#         self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
#         self.finalize_metrics()
#         self.print_results()
#         self.run_callbacks("on_val_end")
#         if self.training:
#             model.float()
#             results = {**stats, **trainer.label_loss_items(self.loss.cpu() / len(self.dataloader), prefix="val")}
#             return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
#         else:
#             LOGGER.info(
#                 "Speed: {:.1f}ms preprocess, {:.1f}ms inference, {:.1f}ms loss, {:.1f}ms postprocess per image".format(
#                     *tuple(self.speed.values())
#                 )
#             )
#             if self.args.save_json and self.jdict:
#                 with open(str(self.save_dir / "predictions.json"), "w", encoding="utf-8") as f:
#                     LOGGER.info(f"Saving {f.name}...")
#                     json.dump(self.jdict, f)  # flatten and save
#                 stats = self.eval_json(stats)  # update stats
#             if self.args.plots or self.args.save_json:
#                 LOGGER.info(f"Results saved to {colorstr('bold', self.save_dir)}")
#             return stats
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#









convert_coco(labels_dir="/Users/yamtawachi/tensorleap/datasets/labels")
coco_yolo_dataset=YOLODataset(img_path=data['path'],data=data, task='detect')
dataloader = build_dataloader(dataset=coco_yolo_dataset,batch=1,workers=0,shuffle=True)

# demo model #
model = YOLO("yolo11n.pt")  # You can use 'yolov11s.pt', 'yolov11m.pt', etc.

for batch in dataloader:
    imgs, labels, bbxs, batch_idx=batch['img']/255.0, batch['cls'], batch['bboxes'], batch['batch_idx']
    predicted=model(imgs)

    annotated_image = predicted[0].plot()

    # Display the image
    plt.imshow(annotated_image)
    plt.axis('off')
    plt.show()
    a=5




# LoadImagesAndVideos(coco_yolo_dataset, batch=1, vid_stride=1)
# dataset = build_yolo_dataset(cfg, data['val'], 1, data, mode="val",stride=32)
# dataloader=build_dataloader(dataset, 1, 0, shuffle=False, rank=-1)
# a=5





