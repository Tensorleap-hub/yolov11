from matplotlib import pyplot as plt

from ultralytics.models.yolo.model import YOLO
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.converter import convert_coco
from ultralytics.data.build import build_dataloader,LoadPilAndNumpy



from config import cfg

#TODO- understand how the cfg in build is created and do the same
dataset_path="coco.yaml"
data =check_det_dataset(dataset_path , autodownload=True)
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





