

# 🚀 Object Detection Using YOLO — Step-by-Step Guide

This guide walks you through integrating YOLO models with Tensorleap for object detection, covering simple usage, pretrained alternatives, and custom-trained models.

---

## 🔰 1. Base Case: YOLOv11s with COCO128

Use this for the simplest setup with the default YOLOv11s model and the COCO128 dataset.

### Steps:

1. **Update Configuration**

   * Open `ultralytics/cfg/default.yaml`
   * Set the following:

     ```yaml
     tensorleap_path: <path_where_tensorleap_is_mounted>
     ```

     This defines where the data and model files will be stored.

2. **Push Model to Tensorleap**

   ```bash
   leap projects push yolov11s.onnx
   ```

   * Downloads necessary files (model/data)
   * Initializes the Tensorleap project

---

## 🧠 2. Using Other Pretrained YOLO Models

To use other YOLO variants from the [Ultralytics suite](https://docs.ultralytics.com/models/):

### Steps:

1. **Edit Configuration**

   * Open `ultralytics/cfg/default.yaml`
   * Update:

     ```yaml
     tensorleap_path: <your_tensorleap_mount_path>
     model: models/<desired_model_name>
     ```

     > ⚠️ Supported: All sizes of YOLOv5, YOLOv8, YOLOv9, YOLOv11, YOLOv12.

2. **Convert `.pt` to ONNX**

   ```bash
   python leap_custom_test.py
   ```

   This will:

   * Convert `.pt` to `.onnx`
   * Print the ONNX path (used in Step 3)
   * Generate `leap_mapping.yaml` file
   * Run a local sanity test on 10 samples

3. **Push to Tensorleap**

   ```bash
   leap projects push <path_to_your_model.onnx>
   ```

   Use the path printed from the previous step.

---

## 🧪 3. Using Your Own Trained YOLO Model

If you’ve trained your own model and/or have custom datasets:

### Steps:

1. **Update Configuration**

   * Open `ultralytics/cfg/default.yaml`
   * Set:

     ```yaml
     tensorleap_path: <your_tensorleap_mount_path>
     model: models/<your_model_name>
     ```
   * Place your `.pt` model in `<tensorleap_path>/models/` and rename it to match the architecture (e.g., `yolov11s.pt`)
   * (Optional) Enable extra dataset support:

     ```yaml
     tensorleap_use_test: True
     tensorleap_use_unlabeled: True
     data: coco.yaml | coco128.yaml | coco8.yaml
     ```

2. **Model Conversion Options**

   **A. If you *do not* have an ONNX model:**

   ```bash
   python leap_custom_test.py
   ```

   * Converts `.pt` to `.onnx`
   * Prints the ONNX path
   * Generates `leap_mapping.yaml`
   * Runs local validation

   **B. If you *already have* an ONNX/H5 model:**

   * Copy it to the root of the repo
   * Find the appropriate `leap_mapping.yaml` file for your YOLO architecture from:

     ```
     ultralytics/tensorleap_folder/mapping/
     ```
   * Copy it to the root and rename to `leap_mapping.yaml`

3. **Push to Tensorleap**

   ```bash
   leap projects push <path_to_your_model.onnx>
   ```

---

## ✅ Summary
| **Use Case**                | **Required Setup**                                     | **Preparation Step**      | **Supported Model Format(s)** |
| --------------------------- |--------------------------------------------------------| ------------------------- | ----------------------------- |
| **Base YOLOv11s + COCO128** | None                                                   | None                      | `onnx`                        |
| **Other Pretrained YOLOs**  | Specify model name in config (`default.yaml`)          | Run `leap_custom_test.py` | `onnx`                        |
| **Custom YOLO (Option A)**  | Place `.pt` model and edit dataset settings            | Run `leap_custom_test.py` | `onnx`                        |
| **Custom YOLO (Option B)**  | Provide `.onnx` or `.h5` model manually + find mapping | None                      | `onnx`, `h5`                  |
