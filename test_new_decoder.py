import os
import yaml
from torch.utils import data
from metric.iou import IoU
import torch.distributed as distrib
# from model.trans4pano_map import Trans4map_segformer
from model.trans4pano_map_new_decoder import Trans4map_segformer

import matplotlib.pyplot as plt
from utils.semantic_utils import color_label

import json
import h5py
import torch
import numpy as np
import torch
import torch.nn
from pathlib import Path
from model.pano_data_loader import DatasetLoader_pano
from torch.utils.data import DistributedSampler
from model.pano_data_loader_show import DatasetLoader_pano_show

from model.model_test import Trans4Map


# data_dir = '/cvhci/data/VisLoc/test_data/'
data_dir = '/cvhci/data/VisLoc/zteng/trans4map_baseline/testing'
output_dir = '/home/zteng/Desktop/Trans4map_outputs/semmap/'

Path(output_dir).mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -- create model
cfg_model = {
    'arch': 'smnet',
    'finetune': False,
    'n_obj_classes': 21,
    'ego_feature_dim': 64,
    'mem_feature_dim': 256,
    'mem_update': 'replace',
    'ego_downsample': False,
}

# model_path = '/cvhci/data/VisLoc/weights_ssmap/smnet_mp3d_best_model_id16301.pkl'
model_path = "/home/zteng/Trans4Map/checkpoints/pano_top_down_mapping/15epoch_train/57028/ckpt_model.pkl"
model_path = "/home/zteng/Desktop/checkpoints/new_decoder/ckpt_model.pkl"

model = Trans4map_segformer(cfg_model, device)
model = model.to(device)

print('Loading pre-trained weights: ', model_path)
# state = torch.load(model_path, map_location='cpu')
state = torch.load(model_path)
print("best_iou:", state['best_iou'])
model_state = state['model_state']
print('model_state:', model_state.keys())


weights = {}
for k, v in model_state.items():
    # if k.startswith('module.rnn') or k.startswith('module.decoder') or k.startswith('module.rnn_r') or k.startswith('module.fuse'):
        # print("key_:", k)
    k = '.'.join(k.split('.')[1:])
    weights[k] = v

model.load_state_dict(weights)

model.eval()


###########################################################################################
config_path = "model/model_pano_decoder.yml"

with open(config_path) as fp:
    cfg = yaml.safe_load(fp)

# test_loader = DatasetLoader_pano(cfg["data"], split=cfg["data"]["test_split"])
test_loader = DatasetLoader_pano_show(cfg["data"], split=cfg["data"]["test_split"])

# test_sampler = DistributedSampler(test_loader, shuffle=False)

testingloader = data.DataLoader(
        test_loader,
        batch_size=1,
        num_workers=cfg["training"]["n_workers"],
        pin_memory=True,
        # sampler=test_sampler,
        multiprocessing_context='fork',
    )

##### setup Metrics #####
obj_running_metrics_test = IoU(cfg['model']['n_obj_classes'])

with torch.no_grad():
    for batch in testingloader:

        # if os.path.isfile(os.path.join(output_dir, env+'.h5')): continue
        rgb, rgb_no_norm, masks_inliers, proj_indices, semmap_gt = batch
        rgb = rgb.to(device)
        proj_indices = proj_indices.to(device)
        masks_inliers = masks_inliers.to(device)

        semmap_gt = semmap_gt.long()

        print('rgb_no_norm:', rgb_no_norm.size())
        # print('masks_inliers:', masks_inliers.size(), torch.sum(masks_inliers==0))

        semmap_pred, observed_masks, rgb_write = model(rgb, proj_indices, masks_inliers, rgb_no_norm)
        # print('semmap_pred_size:', semmap_pred.size())   # torch.Size([1, 21, 500, 500])


        if observed_masks.any():

            semmap_pred = semmap_pred.permute(0,2,3,1)
            print('semmap_pred:', semmap_pred.size())

            semmap_pred_write  = semmap_pred.data.max(-1)[1]

            semmap_mask_write22 = semmap_pred_write

            # semmap_pred_write = semmap_pred_write.squeeze(0)
            semmap_pred_write[~observed_masks] = 0
            semmap_pred_write = semmap_pred_write.squeeze(0)
            print('semmap_pred_write:', semmap_pred_write.size(), semmap_pred_write)




            ############################ semmap projection to show #############################
            semmap_pred_write_out = color_label(semmap_pred_write).squeeze(0)
            print('semmap_pred_write_out:', semmap_pred_write_out.size())
            semmap_pred_write_out = semmap_pred_write_out.permute(1, 2, 0)
            semmap_pred_write_out = semmap_pred_write_out.cpu().numpy().astype(np.uint8)

            ################################################# test on semmap pred_write_out

            plt.rcParams['savefig.dpi'] = 300  # 图片像素
            plt.rcParams['figure.dpi'] = 300  # 分辨率
            #
            plt.subplot(2, 2, 1)
            plt.imshow(semmap_pred_write_out)
            plt.title('Topdown semantic map prediction')
            plt.axis('off')


            ###############################semmap_gt to show ####################################
            semmap_gt_write = semmap_gt.squeeze(0)
            semmap_gt_write_out = color_label(semmap_gt_write).squeeze(0)
            print('semmap_gt:', semmap_gt_write_out.size())
            semmap_gt_write_out = semmap_gt_write_out.permute(1,2,0)
            semmap_gt_write_out = semmap_gt_write_out.cpu().numpy().astype(np.uint8)

            ####################################################################################
            plt.subplot(2, 2, 2)
            plt.imshow(semmap_gt_write_out)
            plt.title('Topdown semantic map gt')
            plt.axis('off')

            ###############################semmap projection mask to show #######################
            # observed_masks_write = observed_masks.squeeze(0)
            semmap_mask_write22[~observed_masks] = 0
            semmap_mask_write22[observed_masks] = 255
            semmap_mask_write22 = semmap_mask_write22.squeeze(0)
            semmap_mask_write22 = semmap_mask_write22.cpu().numpy().astype(np.uint8)

            # plt.subplot(2, 2, 3)
            # plt.imshow(semmap_mask_write22)
            # plt.title('Topdown mask')
            # plt.axis('off')

            rgb_pano_write = rgb_no_norm.squeeze(0)
            rgb_pano_write = rgb_pano_write.cpu().numpy().astype(np.uint8)

            plt.subplot(2, 2, 3)
            plt.imshow(rgb_pano_write)
            plt.title('Pano_image')
            plt.axis('off')


            #####################################################################################

            plt.subplot(2, 2, 4)
            plt.imshow(rgb_write)
            plt.title('Topdown RGB')
            plt.axis('off')

            # plt.show()




            # print('semmap_pred:', semmap_pred.size())  # torch.Size([1, 500, 500, 21])
            masked_semmap_gt = semmap_gt[observed_masks]
            masked_semmap_pred = semmap_pred[observed_masks]
            # print('semmap_pred2:', masked_semmap_pred.size()) # torch.Size([42542, 21])


            obj_gt_val = masked_semmap_gt
            # print('masked_semmap_gt:', masked_semmap_gt.size(), torch.unique(obj_gt_val)) # torch.Size([42542])
            obj_pred_val = masked_semmap_pred.data.max(-1)[1]
            # print('obj_pred_val:', obj_pred_val.size(), masked_semmap_pred.data.max(-1))  # torch.Size([42542])
            obj_running_metrics_test.add(obj_pred_val, obj_gt_val)

conf_metric = obj_running_metrics_test.conf_metric.conf
conf_metric = torch.FloatTensor(conf_metric)
conf_metric = conf_metric.to(device)
# distrib.all_reduce(conf_metric)



conf_metric = conf_metric.cpu().numpy()
conf_metric = conf_metric.astype(np.int32)
tmp_metrics = IoU(cfg['model']['n_obj_classes'])
tmp_metrics.reset()
tmp_metrics.conf_metric.conf = conf_metric
_, mIoU, acc, _, mRecall, _, mPrecision = tmp_metrics.value()

print("val -- mIoU: {}".format(mIoU))
print("val -- mRecall: {}".format(mRecall))
print("val -- mPrecision: {}".format(mPrecision))
print("val -- Overall_Acc: {}".format(acc))


