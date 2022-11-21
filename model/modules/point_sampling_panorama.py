import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.init import normal_



def get_cam_reference_coordinate(reference_points):

    # ref_3d_ = reference_points.cpu()
    ref_3d_ = reference_points

    xss = ref_3d_[:, :, :, 0]
    yss = ref_3d_[:, :, :, 1]
    zss = ref_3d_[:, :, :, 2]

    # X =  depth * np.sin(Theta) * np.cos(Phi)   ##### theta: 0~pi, Phi: 0~2pi
    # Y =  depth * np.sin(Theta) * np.sin(Phi)
    # Z = depth * np.cos(Theta)
    Phi_1 =  torch.arctan(yss/xss)
    Theta_1 = torch.arctan(xss/zss * 1/torch.cos(Phi_1))

    depth = zss/torch.cos(Theta_1)
    depth_absolute = torch.absolute(depth)
    # print('depth:', depth_absolute.min())


    #### 利用cos的单调性
    Theta = torch.arccos(zss/depth_absolute)
    Phi = torch.atan2(yss, xss)

    # print('Phi, Theta:', torch.min(zss/depth) ,Theta.max(), Theta.min(), zss.min(), zss.max())

    Theta = Theta.cpu()
    Phi = Phi.cpu()

    h, w = 512, 1024
    height_num = h* Theta / np.pi - 1/2
    height_num = height_num.ceil()
    width_num = (Phi/np.pi + 1 -1/w) * w/2
    width_num = width_num.ceil()
    ## print('HW:', height_num.size(), width_num.size(), height_num.max(), height_num.min(), width_num.max(), width_num.min())
    # print('HW_num_to_show:', height_num)

    reference_points_cam = torch.cat((height_num, width_num), -1)
    # print('reference_points_cam:', reference_points_cam.size()) ### torch.Size([4, 1, 40000, 2])

    return reference_points_cam



def point_sampling_pano(reference_points, pc_range,  img_metas):
    ##### reference point 和 pc_range 还有变换矩阵换进来, got reference_points_cam and bev_mask

    # lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4) 只是为了延续reference_points tensor的dtype和device

    ## print('lidar2img1:', lidar2img.size(), lidar2img)
    ## torch.Size([1, 6, 4, 4])
    reference_points = reference_points.clone()
    # print("###***:", reference_points.size())
    # print('pc_range:', pc_range)
    ### torch.Size([1, 4, 40000, 3])


    reference_points[..., 0:1] = reference_points[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0]
    reference_points[..., 1:2] = reference_points[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1]
    reference_points[..., 2:3] = reference_points[..., 2:3] * (pc_range[5] - pc_range[2]) + pc_range[2]

    # print('reference_points_why:',[pc_range[5]-pc_range[2]],reference_points[..., 2:3].max(), reference_points[..., 2:3].min())  ### torch.Size([1, 4, 40000, 3])
    ## in Z-direction -1.5~1.5

    # #### 画图 可视化
    # ref_3d_ = reference_points.cpu()
    #
    # fig = plt.figure()
    # ax = fig.add_subplot(projection = '3d')
    # xss = ref_3d_[0, 1, :, 0]
    # yss = ref_3d_[0, 1, :, 1]
    # zss = ref_3d_[0, 1, :, 2]
    #
    # ax.scatter(xss, yss, zss)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    #
    # plt.show()

    reference_points = torch.cat(
        (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    # print('reference_point_0:', reference_points.size()) #### torch.Size([1, 4, 40000, 4]) ##相比之前只是加上了一层全是1的

    reference_points = reference_points.permute(1, 0, 2, 3)
    # print('reference_point_1:', reference_points.size()) #### torch.Size([4, 1, 40000, 4])

    D, B, num_query = reference_points.size()[:3]
    ### num_cam = lidar2img.size(1)     ## 6 cameras
    num_cam = 1

    if num_cam==1:
        reference_points = reference_points.view(D, B, num_query, 4).unsqueeze(-1)
        # print('reference_points_1:', reference_points.size())  ## torch.Size([4, 1, 40000, 4, 1])


    reference_points_cam = get_cam_reference_coordinate(reference_points)
    #### torch.Size([4, 1, 40000, 4])
    ### torch.Size([4, 1, 40000, 2])

    # reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
    ### 矩阵乘法在这里有何作用，只有最后两个维度？坐标转换？把cam转化为...
    # reference_points_cam22 = torch.matmul(lidar2img.to(torch.float32),
    #                                     reference_points.to(torch.float32))
    ### print('reference_points_cam:', reference_points_cam[0,0,0,0,:])  ### torch.Size([4, 1, 6, 40000, 4])
    ### torch.Size([4, 1, 1, 40000, 4])

    # eps = 1e-5
    # bev_mask = (reference_points_cam[..., 2:3] > eps)
    # print('bev_mask_0:', bev_mask.size())  ### torch.Size([4, 1, 6, 40000, 1])
    ### torch.Size([4, 1, 1, 40000, 1])
    ### 经过RT矩阵投影的，z坐标比0大的东西，camera坐标系下
    # print('reference_points_cam[..., 2:3]:', reference_points_cam[..., 2:3])

    # reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
    #     reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    ### torch.Size([4, 1, 1, 40000, 2])

    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][0]  #512
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][1]  #1024

    bev_mask = (  (reference_points_cam[..., 1:2] > 0.0)
                & (reference_points_cam[..., 1:2] < 1.0)
                & (reference_points_cam[..., 0:1] < 1.0)
                & (reference_points_cam[..., 0:1] > 0.0))

    if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        bev_mask = torch.nan_to_num(bev_mask)
    else:
        bev_mask = bev_mask.new_tensor(
            np.nan_to_num(bev_mask.cpu().numpy()))

    reference_points_cam = reference_points_cam.permute(1, 2, 0, 3)
    # print('bev_mask_mask:', bev_mask.size())
    bev_mask = bev_mask.permute(1, 2, 0, 3).squeeze(-1)

    # print('reference_points_cam, bev_as_return:', reference_points_cam.size(), bev_mask.size())
    ### torch.Size([1, 40000, 4, 2]) torch.Size([1, 40000, 4])
    return reference_points_cam, bev_mask

########################################################################################################################
########################################################################################################################
########################################################################################################################

def get_bev_features(
        mlvl_feats, ## 请注意
        bev_queries,
        bev_h,
        bev_w,
        # grid_length=[0.512, 0.512],
        bev_pos=None,
        # prev_bev=None,
        use_cams_embeds = True
        ):
    """
    obtain bev features.
    """
    # print('get_bev_features:', mlvl_feats[1].size(), bev_queries.size(), bev_pos.size())
    ### prev_bev is None
    ### torch.Size([1, 6, 256, 116, 200]) torch.Size([40000, 256]) torch.Size([1, 256, 200, 200])

    bs = mlvl_feats[0].size(0)
    bev_queries = bev_queries.unsqueeze(1).repeat(1, bs, 1)
    bev_pos = bev_pos.flatten(2).permute(2, 0, 1)

    bev_queries = bev_queries.to(device = mlvl_feats[0].device)
    bev_pos = bev_pos.to(device = mlvl_feats[0].device)

    # print('bev_queries, bev_pos:', bev_queries.size(), bev_pos.size())
    ### torch.Size([40000, 1, 256]) torch.Size([40000, 1, 256])


    feat_flatten = []
    spatial_shapes = []


    for lvl, feat in enumerate(mlvl_feats):

        # print('feat_feat0:', feat.size())
        bs, c, h, w = feat.shape
        ## print('hwhw:', h, w) #### 这个mlvl的特征图本来就有4层
        spatial_shape = (h, w)
        feat = feat.flatten(2).permute(0, 2, 1)
        # print('feat_feat1:', feat.size())
        # feat_feat: torch.Size([6, 1, 23200, 256])
        # feat_feat: torch.Size([6, 1, 5800, 256])
        # feat_feat: torch.Size([6, 1, 1450, 256])
        # feat_feat: torch.Size([6, 1, 375, 256])

        # feat_feat: torch.Size([1, 32768, 256])


        if use_cams_embeds:  # True
            num_cams = 1
            embed_dims = 256
            num_feature_levels = 4
            cams_embeds = nn.Parameter(torch.Tensor(num_cams, embed_dims))
            level_embeds = nn.Parameter(torch.Tensor(num_feature_levels, embed_dims))

            normal_(level_embeds)
            normal_(cams_embeds)

            # print('level_embeds:', level_embeds[None, None, lvl:lvl + 1, :].size(), feat.size())
            ### torch.Size([1, 1, 1, 256]) torch.Size([1, 32768, 256])
            # feat = feat + cams_embeds[:, None, None, :].to(feat.dtype)


        level_embeds = level_embeds.to(device = feat.device)
        # print('feat_in_device:', feat.device)
        feat = feat + level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)



        spatial_shapes.append(spatial_shape)
        feat_flatten.append(feat)
        # print('spatial_shapes_feat_flatten:', spatial_shapes, feat_flatten[0].size())
        ### [(116, 200), (58, 100), (29, 50), (15, 25)] spatial_shapes

    feat_flatten = torch.cat(feat_flatten, 2)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
    # print('feat_flatten:', feat_flatten.size())  ### torch.Size([6, 1, 30825, 256])

    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # print('level_start_index_1:', level_start_index.size(), "value_value:", level_start_index)
    ### tensor([0, 23200, 29000, 30450]

    feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims) (6, 30825, 1, 256)

    kwargs = {'img_metas': [{
                             'img_shape': [(928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3), (928, 1600, 3)],
                            }]}

    return bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index
