from ultralytics.cfg import get_cfg
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset
from ultralytics.data.augment import Compose, LetterBox, Format
from ultralytics.data.build import build_dataloader
from ultralytics.models.yolo.model import YOLO
from ultralytics.utils.plotting import Annotator
import matplotlib.pyplot as plt
import numpy as np

# STEP 1: Exactly load data config
cfg = get_cfg(cfg='/Users/yamtawachi/tensorleap/ultralytics/ultralytics/cfg/default.yaml')
cfg.data = 'coco.yaml'
data_info = check_det_dataset(cfg.data, autodownload=True)

# STEP 2: Exactly set transforms used by Ultralytics
transforms = Compose([
    LetterBox(new_shape=(640, 640), auto=False, stride=32),
    Format(bgr=False, normalize=True),
])

# STEP 3: Exactly create dataset like Ultralytics
dataset = YOLODataset(
    img_path=data_info["val"],  # <-- explicitly 'val' or 'train' key from dataset info
    imgsz=640,
    data=data_info,
    augment=transforms
)

# STEP 4: Exactly build dataloader as Ultralytics internally does
dataloader = build_dataloader(
    dataset=dataset,
    batch=1,
    workers=0,
    shuffle=False
)


model = YOLO("tensorleap_folder/yolo11n.pt")

for batch in dataloader:
    imgs = batch['img']/255.
    labels = batch['cls']
    bboxes = batch['bboxes']
    paths = batch['im_file']

    # Run inference
    predicted = model(imgs)

    # Original image (unnormalized for visualization)
    orig_img = predicted[0].orig_img.copy()

    # Plot ground truth using Ultralytics built-in Annotator
    annotator_gt = Annotator(np.ascontiguousarray(orig_img))
    for cls, bbox in zip(labels[0], bboxes[0]):
        label_name = dataset.data['names'][int(cls)]
        annotator_gt.box_label(bbox, label_name, color=(0, 255, 0))
    gt_img = annotator_gt.result()

    # Plot predictions (built-in)
    pred_img = predicted[0].plot()

    # Display side-by-side
    plt.figure(figsize=(15, 10))

    # Ground Truth
    plt.subplot(1, 2, 1)
    plt.imshow(gt_img)
    plt.title("Ground Truth (Ultralytics-style)")
    plt.axis('off')

    # Prediction
    plt.subplot(1, 2, 2)
    plt.imshow(pred_img)
    plt.title("Predicted (Ultralytics built-in)")
    plt.axis('off')

    plt.show()
