from pathlib import Path
from types import SimpleNamespace
import yaml
from ultralytics import YOLO
import re


def yolo_version_check(model_path):
    model_name = model_path.stem
    match = re.fullmatch(r'yolov5([a-z])', model_name)
    exported_path= model_path.with_name(model_name + 'u' + '.onnx') if match else model_path.with_suffix('.onnx')
    if not exported_path.exists():
        raise FileNotFoundError(f"File {exported_path} not found, check the name of the yolo exported version.")
    return exported_path


def dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(i) for i in d]
    else:
        return d
def export_to_h5(onnx_path):
    from onnx2keras import onnx_to_keras
    from onnx2keras import convert_channels_first_to_last
    from keras_data_format_converter.converterapi import convert_channels_first_to_last

    import onnx

    onnx_model = onnx.load(onnx_path)
    input_all = [_input.name for _input in onnx_model.graph.input]
    input_initializer = [node.name for node in onnx_model.graph.initializer]
    input_names = list(set(input_all) - set(input_initializer))

    k_model = onnx_to_keras(onnx_model, input_names, name_policy='attach_weights_name', allow_partial_compilation=False)
    flipped_model = convert_channels_first_to_last(k_model.converted_model, should_transform_inputs_and_outputs=False)
    flipped_model.save(Path(onnx_path).with_suffix('.h5'))
    print(f"Model exported to h5: {Path(onnx_path).with_suffix('.h5')}")


def export_to_onnx(cfg):
    model = YOLO(cfg.model if hasattr(cfg, "model") else "yolo11s.pt")
    model.export(format="onnx", nms=False, export_train_head=True)

def start_export(h5=False):
    file_path = 'ultralytics/cfg/default.yaml'
    with open(file_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    cfg=dict_to_namespace(config_dict)

    model_path=Path(cfg.model)
    if not model_path.is_absolute():
        model_path = (cfg.tensorleap_path / model_path).resolve()
    assert model_path.is_relative_to(cfg.tensorleap_path), (
        f"❌ {model_path!r} is not inside tensorleap path {cfg.tensorleap_path!r}" )
    cfg.model=model_path
    export_to_onnx(cfg)
    exported_model_path=yolo_version_check(model_path)
    if exported_model_path.exists() and h5:
        export_to_h5(exported_model_path)

    print(f"Model exported to ONNX: {exported_model_path}")
    return str(exported_model_path)
if __name__ == '__main__':
    start_export()