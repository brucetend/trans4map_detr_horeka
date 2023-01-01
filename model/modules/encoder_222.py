
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

#from projects.mmdet3d_plugin.models.utils.bricks import run_time
import sys, os
cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
root_path = os.path.split(root_path)[0]
sys.path.append(root_path)
from model.modules.utils import position_embedding
print('haha:',root_path, cur_path)
from utils.visual import save_tensor
###########to import everything in Folder modules
from point_sampling_panorama import get_bev_features
from resnet_mmcv import *

from mmcv.cnn.bricks.transformer import build_positional_encoding



# from custom_base_transformer_layer import MyCustomBaseTransformerLayer
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)

from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
from mmcv.runner.base_module import BaseModule

import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])



########################################################################################################################
if __name__ == '__main__':
    ### arguments to input

    kwargs = {'img_metas': [{
                             'img_shape': [(512, 1024, 3), (512, 1024, 3), (512, 1024, 3), (512, 1024, 3), (512, 1024, 3), (512, 1024, 3)],
                            }]}


    ### 定义模型 num_layers 改成 2.
    # encoder = {'type': 'BEVFormerEncoder', 'num_layers': 2, 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'num_points_in_pillar': 4, 'return_intermediate': False, 'transformerlayers': {'type': 'BEVFormerLayer', 'attn_cfgs': [{'type': 'TemporalSelfAttention', 'embed_dims': 256, 'num_levels': 1}, {'type': 'SpatialCrossAttention', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0], 'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4}, 'embed_dims': 256}], 'feedforward_channels': 512, 'ffn_dropout': 0.1, 'operation_order': ('cross_attn', 'norm', 'ffn', 'norm')}}
    encoder = {'type': 'BEVFormerEncoder',
               'num_layers': 2,
               'pc_range': [-5, -5, -2, 5, 5, 1], # pc_range: pointcloud_range_XYZ
               'num_points_in_pillar': 4,
               'return_intermediate': False,
               'transformerlayers': {'type': 'BEVFormerLayer',
                                     'attn_cfgs': [{'type': 'SpatialCrossAttention', 'pc_range': [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
                                                    'deformable_attention': {'type': 'MSDeformableAttention3D', 'embed_dims': 256, 'num_points': 8, 'num_levels': 4}, 'embed_dims': 256}],
                                     'feedforward_channels': 512,
                                     'ffn_dropout': 0.1,
                                     'operation_order': ('cross_attn', 'norm', 'ffn', 'norm')}}

    nice_encoder = build_transformer_layer_sequence(encoder)

    print('nice_encoder:', nice_encoder)

    device = "cuda"
########################################################################################################################
    ################# Backbone #####################
    Encoder_mmcv = ResNet(depth=101).to(device)
    print('encoder_mmcv:', Encoder_mmcv)

    img = torch.randn([1, 3, 1024, 2048]).to(device)
    ml_feat = Encoder_mmcv(img)
    # print('ml_feat:', len(ml_feat),ml_feat[0].size(), ml_feat[1].size(), ml_feat[2].size(), ml_feat[3].size())

    in_channels = [256, 512, 1024, 2048]
    fpn_mmdet = FPN(in_channels, 256, len(in_channels)).eval().to(device)
    feat_fpn = fpn_mmdet(ml_feat)

    # feat_fpn_flatten_list = []
    # for i in range(4):
    #     feat_fpn_flatten = feat_fpn[i].flatten(start_dim= 2)
    #     feat_fpn_flatten = feat_fpn_flatten.permute(0,2,1)
    #     feat_fpn_flatten_list.append(feat_fpn_flatten)

    print('feat_fpn:', len(feat_fpn), feat_fpn[0].size(), feat_fpn[1].size(), feat_fpn[2].size(), feat_fpn[3].size())


########################################################################################################################
    height, width = 250, 250
    bev_h, bev_w = height, width
    embed_dims = 256
    bs = 1

    mlvl_feats = feat_fpn
    dtype = mlvl_feats[0].dtype
    print('dtype:', dtype)


    bev_bev_embedding = nn.Embedding(bev_h * bev_w, embed_dims)
    bev_queries = bev_bev_embedding.weight.to(dtype)

    positional_encoding = dict(type='SinePositionalEncoding',
                               num_feats=128,
                               normalize=True)
    positional_encoding_bev = build_positional_encoding(positional_encoding)

    ### 初始化的 bev_queries

    bev_mask = torch.zeros((bs, bev_h, bev_w), device=bev_queries.device).to(dtype)
    bev_pos = positional_encoding_bev(bev_mask).to(dtype)
    print('bev_pos:', bev_pos.size())

    bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index = get_bev_features(mlvl_feats, bev_queries, bev_h, bev_w, bev_pos)

    # print('device_haha:', bev_queries.device, feat_flatten.device, bev_pos.device, spatial_shapes.device, level_start_index.device)
########################################################################################################################

    # a = torch.randn([1, 1, 32768, 256])
    # b = torch.randn([1, 1, 8192, 256])
    # c = torch.randn([1, 1, 2048, 256])
    # d = torch.randn([1, 1, 512,  256])
    #
    # feat_flatten = [a, b,c, d]

    # feat_flatten = torch.cat(feat_flatten, 2).permute(0,2,1,3).to(device)
    # bev_h = 200
    # bev_w = 200
    # print('feat_flatten_in_test:', feat_flatten.size())  ### torch.Size([6, 42624, 1, 256])

    ####################################################################################################################
    # spatial_shapes = torch.tensor([[128, 256],
    #                                [ 64, 128],
    #                                [ 32,  64],
    #                                [ 16,  32]], device='cuda:0')
    #
    # level_start_index = torch.tensor([0, 32768, 40960, 43008], device='cuda:0')

    # prev_bev = torch.randn([40000, 1, 256], device= 'cuda:0')
    prev_bev = None
    # shift = torch.tensor([[-0.0001,  0.0416]], device='cuda:0')
    shift = None
    ##### 这两个值，对于无temporal attention 的模型来说，木有意义！


    nice_encoder = nice_encoder.to(device)
    bev_embed = nice_encoder(
        bev_queries,
        feat_flatten,                   ##### 四层feature map 拉直了来的，降采样8
        feat_flatten,
        bev_h=bev_h,
        bev_w=bev_w,
        bev_pos=bev_pos,
        spatial_shapes=spatial_shapes,  ##### 都是feature map里来的
        level_start_index=level_start_index,
        prev_bev=prev_bev,
        shift=shift,
        **kwargs
    )

    print("bev_embed:", bev_embed.size())
