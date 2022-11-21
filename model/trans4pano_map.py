import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as transforms
from Backbone.segformer import Segformer
from torchsummary import summary
from imageio import imwrite
import matplotlib.pyplot as plt

# from mmcv.cnn import ConvModule


normalize = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])

depth_normalize = transforms.Normalize(mean=[0.213], std=[0.285])

map_width = 500


class Trans4map_segformer(nn.Module):
    def __init__(self, cfg, device):
        super(Trans4map_segformer, self).__init__()

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

        if mem_update == 'replace':
            self.linlayer = nn.Linear(ego_feat_dim, mem_feat_dim)

        ########################################### segformer and decoder ##############################################
        self.encoder = Segformer()

        self.pretrained_model_path = "/home/zteng/Trans4Map/checkpoints/mit_b2.pth"
        # load pretrained weights
        state = torch.load(self.pretrained_model_path)
        #print('state:', state.keys())
        weights = {}
        for k, v in state.items():
            # print('key_:', k)
            weights[k] = v

        self.encoder.load_state_dict(weights, strict=False)

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

    def memory_update(self, features, proj_indices, masks_inliers, rgb_features):

        features = features.float() # torch.Size([1, 1, 64, 256, 512])

        N, T, C, H, W = features.shape
        # T = 1, N = 1


        if self.mem_update == 'replace':

            state = torch.zeros((N * map_width * map_width, self.mem_feat_dim), dtype=torch.float, device=self.device_mem)
            state_rgb = torch.zeros((N * map_width * map_width, 3), dtype=torch.uint8, device=self.device_mem)

        observed_masks = torch.zeros((N, map_width, map_width), dtype=torch.bool, device=self.device)

        ################################################################################################################
        # print('feature:', features.size())
        feature = features[:, 0, :, :, :].to(self.device)    # torch.Size([1, 64, 128, 256])

        # print('**mask_inliers:', masks_inliers.size())
        mask_inliers = masks_inliers[:, :, :]                # torch.Size([1, 128, 256])

        # print('proj_index:', proj_indices.size())
        proj_index = proj_indices                            # torch.Size([1, 25000])
        #### how to fill these TO DO!

        # m = (proj_index >= 0)  # -- (N, 500*500)
        threshold_index_m = torch.max(proj_index).item()
        m = (proj_index < threshold_index_m)

        if N > 1:
            batch_offset = torch.zeros(N, device=self.device)
            batch_offset[1:] = torch.cumsum(mask_inliers.sum(dim=1).sum(dim=1), dim=0)[:-1]
            batch_offset = batch_offset.unsqueeze(1).repeat(1, map_width * map_width).long()
            proj_index += batch_offset


        if m.any():
            feature = F.interpolate(feature, size=(1024, 2048), mode="bilinear", align_corners=True)
            if self.ego_downsample:
                feature = feature[:, :, ::4, ::4]

            feature = feature.permute(0, 2, 3, 1)  # -- (N,H,W,512) # torch.Size([1, 480, 640, 64])

            feature = feature[mask_inliers, :]     # torch.Size([841877, 64])
            # print('feature_segformer:', feature.size())

            tmp_memory = feature[proj_index[m], :] # torch.Size([112116, 64])
            print('tmp_memory:', tmp_memory.size())


            # rgb_features = rgb_features.squeeze(0)
            # print('size_of_rgb_features:', rgb_features.size())

            # rgb_features = rgb_features.permute(0, 2, 3, 1)
            rgb_features = rgb_features[mask_inliers, :]
            rgb_memory = rgb_features[proj_index[m], :]
            # print('rgb_memory:', rgb_memory.size(), rgb_memory)


            # print('m_view:', m.shape)
            tmp_top_down_mask = m.view(-1)         # torch.Size([250000])
            # print('tmp_top_down_mask***:', torch.sum(tmp_top_down_mask!=0))

            if self.mem_update == 'replace':
                tmp_memory = self.linlayer(tmp_memory)
                # print("tmp_memory_size:", tmp_memory.size())

                state[tmp_top_down_mask, :] = tmp_memory.to(self.device_mem)  ### torch.size([250000, 256])

                ### state_rgb[tmp_top_down_mask, :] = (rgb_memory * 255).to(self.device_mem)
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

            else:
                raise NotImplementedError

            ############################################################################################################
            observed_masks += m.reshape(N, map_width, map_width)   # torch.Size([1, 500, 500])
            print('observed_masks:', torch.sum(observed_masks==0), observed_masks.size())

            del tmp_memory
        del feature


        if self.mem_update == 'replace':
            memory = state

        memory = memory.view(N, map_width, map_width, self.mem_feat_dim) # torch.Size([1, 250, 250, 256])

        memory = memory.permute(0, 3, 1, 2) # torch.Size([1, 256, 250, 250])
        # print('memory_size:', memory.size())

        # memory = self.fuse(memory)
        memory = memory.to(self.device)
        observed_masks = observed_masks.to(self.device)

        return memory, observed_masks, rgb_write


    # def forward(self, features, proj_indices, masks_inliers):
    def forward(self, rgb, proj_indices, masks_inliers, rgb_no_norm):
        # print('rgb_rgb:', rgb.size())

        rgb_features = torch.nn.functional.interpolate(rgb, size=(3, 512, 1024), mode = 'nearest', align_corners=None)
        # print('shape_features:', features.size())

        #summary(self.encoder, (1, 3, 512, 1024))
        #print(summary)

        features = self.encoder(rgb_features)     # torch.Size([1, 1, 3, 512, 1024])


        features = features.unsqueeze(0)      # torch.Size([1, 1, 64, 128, 256])
        # predictions = F.interpolate(predictions, size=(480,640), mode="bilinear", align_corners=True)
        memory, observed_masks, rgb_write = self.memory_update(features,
                                                    proj_indices,
                                                    masks_inliers,
                                                    rgb_no_norm)

        semmap = self.decoder(memory)
        return semmap, observed_masks, rgb_write
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