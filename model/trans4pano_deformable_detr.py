import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
# from Backbone.segformer import Segformer
from Backbone.resnet_mmcv import ResNet
from torchsummary import summary
from imageio import imwrite
import matplotlib.pyplot as plt
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from mmdet.models.necks import FPN
from mmcv.cnn.bricks.transformer import build_positional_encoding
from model.modules.point_sampling_panorama import get_bev_features




normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])

map_width = 500

class Trans4map_deformable_detr(nn.Module):
    def __init__(self, cfg, device):
        super(Trans4map_deformable_detr, self).__init__()

        ego_feat_dim = cfg['ego_feature_dim']
        mem_feat_dim = cfg['mem_feature_dim']
        n_obj_classes = cfg['n_obj_classes']

        mem_update = cfg['mem_update']
        ego_downsample = cfg['ego_downsample']

        self.mem_feat_dim = mem_feat_dim
        self.mem_update = mem_update
        self.ego_downsample = ego_downsample
        self.device = device
        self.device_mem = device  # cpu
        # self.device_mem = torch.device('cuda')  # cpu

        ################################################################################################################
        #### 新增 encoding 初始化！

        self.bev_h = 250
        self.bev_w = 250
        self.embed_dims = 256
        self.bs = 1

        bev_bev_embedding = nn.Embedding(self.bev_h * self.bev_w, self.embed_dims)
        dtype = torch.float32
        self.bev_queries = bev_bev_embedding.weight.to(dtype)

        positional_encoding = dict(type='SinePositionalEncoding',
                                   num_feats=128,
                                   normalize=True)
        positional_encoding_bev = build_positional_encoding(positional_encoding)

        self.bev_mask = torch.zeros((self.bs, self.bev_h, self.bev_w), device= self.bev_queries.device).to(dtype)
        self.bev_pos = positional_encoding_bev(self.bev_mask).to(dtype)

        ################################################################################################################

        self.encoder_backbone = ResNet(depth = 101)

        self.encoder_cfg = {'type': 'BEVFormerEncoder',
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
        self.encoder = build_transformer_layer_sequence(self.encoder_cfg )


        if mem_update == 'replace':
            self.linlayer = nn.Linear(ego_feat_dim, mem_feat_dim)

        ########################################### segformer and decoder ##############################################
        # self.encoder = Segformer()


        # self.pretrained_model_path = "/home/zteng/Trans4Map/checkpoints/mit_b2.pth"
        # # load pretrained weights
        # state = torch.load(self.pretrained_model_path)
        # #print('state:', state.keys())
        # weights = {}
        # for k, v in state.items():
        #     # print('key_:', k)
        #     weights[k] = v

        # self.encoder.load_state_dict(weights, strict=False)

        # self.fuse = nn.Conv2d(mem_feat_dim*2, mem_feat_dim, 1, 1, 0)

        self.decoder = Decoder(mem_feat_dim, n_obj_classes)

    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.fill_(1.)
            m.bias.data.fill_(1e-4)

    def mask_update(self,  # features,
                    proj_indices, masks_inliers, rgb_features):

        # features = features.float() # torch.Size([1, 1, 64, 256, 512])

        # N, T, C, H, W = features.shape
        # # T = 1, N = 1
        bs = 1


        # if self.mem_update == 'replace':
        #
        #     state = torch.zeros((N * map_width * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
        #     state_rgb = torch.zeros((N * map_width * map_width, 3), dtype=torch.uint8, device=self.device_mem)
        state_rgb = torch.zeros((bs * map_width * map_width, 3), dtype=torch.uint8, device=self.device_mem)

        observed_masks = torch.zeros((bs, map_width, map_width), dtype=torch.bool, device=self.device)

        ################################################################################################################
        # # print('feature:', features.size())
        # feature = features[:, 0, :, :, :].to(self.device)    # torch.Size([1, 64, 128, 256])

        # # print('**mask_inliers:', masks_inliers.size())
        mask_inliers = masks_inliers[:, :, :]                # torch.Size([1, 128, 256])

        # # print('proj_index:', proj_indices.size())
        proj_index = proj_indices                            # torch.Size([1, 250000])
        #### how to fill these TO DO!

        # m = (proj_index >= 0)  # -- (N, 500*500)
        threshold_index_m = torch.max(proj_index).item()
        m = (proj_index < threshold_index_m)

        # if N > 1:
        #     batch_offset = torch.zeros(N, device=self.device)
        #     batch_offset[1:] = torch.cumsum(mask_inliers.sum(dim=1).sum(dim=1), dim=0)[:-1]
        #     batch_offset = batch_offset.unsqueeze(1).repeat(1, map_width * map_width).long()
        #     proj_index += batch_offset


        if m.any():
            # feature = F.interpolate(feature, size=(1024, 2048), mode="bilinear", align_corners=True)
            # if self.ego_downsample:
            #     feature = feature[:, :, ::4, ::4]

            # feature = feature.permute(0, 2, 3, 1)  # -- (N,H,W,512) # torch.Size([1, 480, 640, 64])

            # feature = feature[mask_inliers, :]     # torch.Size([841877, 64])
            # print('feature_segformer:', feature.size())

            # tmp_memory = feature[proj_index[m], :] # torch.Size([112116, 64])
            # print('tmp_memory:', tmp_memory.size())


            # # rgb_features = rgb_features.squeeze(0)
            # # print('size_of_rgb_features:', rgb_features.size())

            # rgb_features = rgb_features.permute(0, 2, 3, 1)
            rgb_features = rgb_features[mask_inliers, :]
            rgb_memory = rgb_features[proj_index[m], :]
            # print('rgb_memory:', rgb_memory.size(), rgb_memory)


            # print('m_view:', m.shape)
            tmp_top_down_mask = m.view(-1)         # torch.Size([250000])
            # print('tmp_top_down_mask***:', torch.sum(tmp_top_down_mask!=0))

            # if self.mem_update == 'replace':
            #     tmp_memory = self.linlayer(tmp_memory)
            #     # print("tmp_memory_size:", tmp_memory.size())
            #
            #     state[tmp_top_down_mask, :] = tmp_memory.to(self.device_mem)  ### torch.size([250000, 256])
            #
            #     ### state_rgb[tmp_top_down_mask, :] = (rgb_memory * 255).to(self.device_mem)
            #     state_rgb[tmp_top_down_mask, :] = rgb_memory.to(self.device_mem)
            state_rgb[tmp_top_down_mask, :] = rgb_memory.to(self.device_mem)

            ############################ rgb projection to show #############################
            rgb_write = torch.reshape(state_rgb,(500, 500, 3))
            # print('state_rgb:', state_rgb.size(), rgb_write.size())

            rgb_write = rgb_write.cpu().numpy().astype(np.uint8)
                #
                # plt.imshow(rgb_write)
                # plt.title('Topdown semantic map prediction')
                # plt.axis('off')
                # plt.show()

            # else:
            #     raise NotImplementedError

            ############################################################################################################
            observed_masks += m.reshape(bs, map_width, map_width)   # torch.Size([1, 500, 500])
            # print('observed_masks:', torch.sum(observed_masks==0), observed_masks.size())

            # del tmp_memory
        # del feature

        # if self.mem_update == 'replace':
        #     memory = state

        # memory = memory.view(N, map_width, map_width, self.mem_feat_dim) # torch.Size([1, 250, 250, 256])

        # memory = memory.permute(0, 3, 1, 2) # torch.Size([1, 256, 250, 250])
        # # print('memory_size:', memory.size())

        # # memory = self.fuse(memory)
        # memory = memory.to(self.device)

        observed_masks = observed_masks.to(self.device)

        return observed_masks, rgb_write


    # def forward(self, features, proj_indices, masks_inliers):
    def forward(self, rgb, proj_indices, masks_inliers, rgb_no_norm):
        # print('rgb_rgb:', rgb.size())

        #rgb_features = torch.nn.functional.interpolate(rgb, size=(3, 512, 1024), mode = 'nearest', align_corners=None)
        #rgb_features = rgb_features.squeeze(0)
        # print('shape_features:', rgb_features.size())
        rgb_features = rgb.squeeze(0)

        #summary(self.encoder, (1, 3, 512, 1024))
        #print(summary)

        # features = self.encoder(rgb_features)     # torch.Size([1, 1, 3, 512, 1024])
        ml_feat = self.encoder_backbone(rgb_features)

        in_channels = [256, 512, 1024, 2048]
        fpn_mmdet = FPN(in_channels, 256, len(in_channels)).eval()
        fpn_mmdet = fpn_mmdet.to(device = "cuda")
        feat_fpn = fpn_mmdet(ml_feat)

        bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index = get_bev_features(
            feat_fpn, self.bev_queries, self.bev_h, self.bev_w, self.bev_pos)
        prev_bev = None
        # shift = torch.tensor([[-0.0001,  0.0416]], device='cuda:0')
        shift = None
        kwargs = {'img_metas': [{
            'img_shape': [(1024, 2048, 3)],
        }]}


        bev_embed = self.encoder(
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

        # features = features.unsqueeze(0)      # torch.Size([1, 1, 64, 128, 256])
        # predictions = F.interpolate(predictions, size=(480,640), mode="bilinear", align_corners=True)

        observed_masks, rgb_write = self.mask_update(
                                                    proj_indices,
                                                    masks_inliers,
                                                    rgb_no_norm)

        # print('bev_embed:', bev_embed.size())
        ###改变位置
        bs = 1
        bev_embed = bev_embed.permute(0, 2, 1)
        bev_embed = bev_embed.view(bs, 256, self.bev_h, self.bev_w)

        ##### 特征尺寸无法这么搞！！！
        bev_embed = F.interpolate(bev_embed, size=(500, 500), mode="bilinear", align_corners=True)

        # semmap = self.decoder(memory)
        semmap = self.decoder(bev_embed)


        # return semmap, observed_masks, rgb_write
        return semmap, observed_masks
        ## return memory, observed_masks


class Decoder(nn.Module):
    def __init__(self, feat_dim, n_obj_classes):
        super(Decoder, self).__init__()

        self.layer = nn.Sequential(nn.Conv2d(feat_dim, 128, kernel_size=7, stride=1, padding=3, bias=False),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(64),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(64, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU(inplace=True),
                                   )

        self.obj_layer = nn.Sequential(nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1, bias=False),
                                       nn.BatchNorm2d(48),
                                       nn.ReLU(inplace=True),
                                       nn.Conv2d(48, n_obj_classes,
                                                 kernel_size=1, stride=1,
                                                 padding=0, bias=True),
                                       )

    def forward(self, memory):
        # print("memory_shape:", memory.size())
        l1 = self.layer(memory)
        out_obj = self.obj_layer(l1)
        # print("out_obj_shape:", out_obj.size())
        return out_obj