# import torch
import torch
from charset_normalizer import detect

from leap_binder import (input_encoder, preprocess_func_leap, gt_encoder,
                         leap_binder, dummy_loss, metadata_sample_index, gt_bb_decoder, image_visualizer, bb_decoder)
import tensorflow as tf
import numpy as np
from code_loader.helpers import visualize
from code_loader.helpers.detection.yolo.utils import reshape_output_list
from ultralytics.tensorleap_folder.config import cfg



def check_custom_test():
    check_generic = True
    plot_vis= True
    if check_generic:
        leap_binder.check()
    print("started custom tests")

    # load the model
    model_path = r"yolov11s.h5"
    model = tf.keras.models.load_model(model_path)

    from ultralytics import YOLO
    model_base = YOLO("yolo11s.pt")
    model_base.args=cfg
    model_base.model.model[-1].reg_max=1

    responses = preprocess_func_leap()
    for subset in responses:
        for idx in range(10):
            image = input_encoder(idx, subset)
            concat = np.expand_dims(image, axis=0)
            y_pred = model([concat]).numpy()
            gt = gt_encoder(idx, subset)
###################### loss #########################
            cls_ls,bbx_ls= reshape_output_list(tf.convert_to_tensor(y_pred.transpose(0,2,1)),decoded=True,image_size=640, priors=1)
            pred_ls=[]
            scales_ls=[(80,80),(40,40),(20,20)]
            for i in range(len(cls_ls)):
                pred_ls.append(torch.concatenate([torch.from_numpy(bbx_ls[i].numpy()), torch.from_numpy(cls_ls[i].numpy())], dim=2).permute(0, 2, 1).reshape(1,84,*scales_ls[i]) )
            d=subset.data['dataloader'].labels[idx]
            d["bboxes"]=torch.from_numpy(d["bboxes"])
            d["cls"]=torch.from_numpy(d["cls"])
            d["batch_idx"]=torch.zeros_like(d['cls'])
            criterion =model_base.init_criterion()
            # criterion.no=84 # why?
            # criterion.reg_max=1 #why?
            # criterion.proj=torch.ones(1)
            from ultralytics.utils import IterableSimpleNamespace
            criterion.hyp = IterableSimpleNamespace(**criterion.hyp)
            criterion.hyp.box=7.5
            criterion.hyp.cls = 0.5
            criterion.hyp.dfl= 1.5
            L= criterion(pred_ls, d)


#####################################################
            img_vis=image_visualizer(image)
            gt_img=gt_bb_decoder(image,gt)
            pred_img=bb_decoder(image,y_pred)

            if plot_vis:
                # visualize(img_vis)
                # visualize(gt_img)
                visualize(pred_img)


            d_loss=dummy_loss(y_pred,gt)
            metadata_sample=metadata_sample_index(idx,subset)
    print("finish tests")


if __name__ == '__main__':
    check_custom_test()
