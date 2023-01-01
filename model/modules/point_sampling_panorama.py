import numpy as np
import torch
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.nn.init import normal_
import cv2
import h5py



def get_reference_points(H, W, map_heights, map_mask, bs=1, device='cuda', dtype=torch.float, ):

    row_column_index = np.where(map_mask == True)
    row = row_column_index[0]
    column = row_column_index[1]

    x_pos = row * 0.02 - 0.01 - 5
    y_pos = (H - column) * 0.02 + 0.01 - 5
    z_pos = map_heights[row, column] - 10.0

    real_position = np.stack([x_pos, y_pos, z_pos], axis=1)
    ref_3d = np.array([[real_position]])
    # print('real_position:', ref_3d.shape, real_position[:, 2].min(), real_position[:, 2].max())
    # (1, 1, 32424, 3)
    return ref_3d


def get_cam_reference_coordinate(reference_points, height, width):
# def get_cam_reference_coordinate(reference_points, height, width, img, mask):

    # ref_3d_ = reference_points.cpu()
    ref_3d_ = reference_points

    xss = ref_3d_[:,:,:, 0]
    yss = ref_3d_[:,:,:, 1]
    zss = ref_3d_[:,:,:, 2]

    #### plot reference_points ###
    # ref_3d_plot = ref_3d_
    # fig = plt.figure()
    # ax = fig.add_subplot(projection='3d')
    # xss_plot = ref_3d_plot[:, 0]
    # yss_plot = ref_3d_plot[:, 1]
    # zss_plot = ref_3d_plot[:, 2]
    #
    # ax.scatter(xss_plot, yss_plot, zss_plot)
    #
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # plt.show()

    ####################################################################################################################
    # X =  depth * np.sin(Theta) * np.cos(Phi)   ##### theta: 0~pi, Phi: 0~2pi
    # Y =  depth * np.sin(Theta) * np.sin(Phi)
    # Z = depth * np.cos(Theta)

    # xss = torch.from_numpy(xss)
    # yss = torch.from_numpy(yss)
    # zss = torch.from_numpy(zss)


    Phi_1 =  torch.atan2(yss, xss)
    # Phi_2 = torch.arctan(yss/xss)
    # print('Phi_1:', Phi_1.max(), Phi_1.min())
    Theta_1 = torch.arctan( xss/zss * 1/torch.cos(Phi_1))

    depth = zss/torch.cos(Theta_1)
    depth_absolute = torch.absolute(depth)

    # print('depth:', depth_absolute.min())

    #### 利用cos的单调性
    Theta = torch.arccos(zss/depth_absolute)
    Phi = -torch.atan2(yss, xss)
    # Phi = np.pi - torch.arctan2(yss, xss) + np.pi/2
    # Phi[Phi > np.pi * 2] -= np.pi * 2

    ## print('Phi:', Phi.max(), Phi.min())
    ## print('Phi, Theta:', torch.min(zss/depth) ,Theta.max(), Theta.min(), zss.min(), zss.max())

    Theta = Theta.cpu()
    Phi = Phi.cpu()

    # h, w = 1034, 2048
    h,w = height, width

    height_num = h * Theta / np.pi
    # height_num = h * (1- Theta / np.pi)
    height_num = height_num.ceil()


    # width_num = (Phi/np.pi + 1 - 1/w) * w/2
    width_num = (Phi/np.pi - 1/w) * w/2
    width_num[width_num < 0] += 2048

    width_num = width_num.ceil()
    # print('HW:', height_num.size(), width_num.size(), height_num.max(), height_num.min(), width_num.max(), width_num.min())
    # print('HW_num_to_show:', height_num)

    ##### histogram height_num
    # hist, bins = np.histogram(width_num, bins = 100, range = (1, 2048))
    # print('hist:', hist, 'bins:', bins)

    height_num = height_num.unsqueeze(-1)
    width_num = width_num.unsqueeze(-1)
    reference_points_cam = torch.cat((height_num, width_num), 3)
    # print('reference_points_cam00:', reference_points_cam[...,1].max(), reference_points_cam[...,1].min()) ### torch.Size([4, 1, 40000, 2])

    #
    # ############################### vis_vis ##############################
    # map_map = reference_points_cam.numpy().astype(np.int32)
    # # print('map_map:', map_map[:, 1].max())
    #
    # mask = torch.from_numpy(mask)
    # mask_flatten = torch.flatten(mask)
    #
    # bev_index_haha = 0
    # pixelwise_feat = []
    # for i in range(250000):
    #     # bev_index_haha = i
    #     # print('bev_index_haha:', i, map_map.shape)
    #
    #     if mask_flatten[i] == False:
    #         pixel_value = [0,0,0]
    #     elif mask_flatten[i] == True:
    #         u, v = map_map[0,0,bev_index_haha,0], map_map[0,0,bev_index_haha, 1]
    #         # print('u,v:', u, v)
    #         pixel_value = img[u, v, :]
    #         bev_index_haha = bev_index_haha + 1
    #         # print('pixel_value:', pixel_value)
    #
    #     pixelwise_feat.append(pixel_value)
    #
    #
    # bev_haha = np.array(pixelwise_feat)
    # bev_haha = bev_haha.reshape(500,500,3)
    # print('bev_haha:', bev_haha)
    #
    # plt.imshow(bev_haha)
    # # plt.imshow(img)
    # plt.title('Topdown semantic map prediction')
    # plt.axis('off')
    # plt.show()

    return reference_points_cam



def point_sampling_pano(ref_3d,  pc_range,  img_metas, map_mask):
    ##### reference point 和 pc_range 还有变换矩阵换进来, got reference_points_cam and bev_mask


    # print('pc_range:', pc_range)
    ### torch.Size([1, 4, 40000, 3])

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

    # print('reference_point_1:', reference_points.size()) #### torch.Size([4, 1, 40000, 4])

    ####################################################################################################################
    #### size of input img ####
    img_height = img_metas[0]['img_shape'][0][0]
    img_width = img_metas[0]['img_shape'][0][1]

    reference_points_cam = get_cam_reference_coordinate(ref_3d, img_height, img_width)
    #### torch.Size([4, 1, 40000, 4])
    ### torch.Size([4, 1, 40000, 2])



    reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][0]  #1024
    reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][1]  #2048

    # print('reference_points_cam:', reference_points_cam[..., 0].max(), reference_points_cam[...,1].min())


    # bev_mask 是现成的
    # bev_mask = (  (reference_points_cam[..., 1:2] > 0.0)
    #             & (reference_points_cam[..., 1:2] < 1.0)
    #             & (reference_points_cam[..., 0:1] < 1.0)
    #             & (reference_points_cam[..., 0:1] > 0.0))
    bev_mask = map_mask

    # if digit_version(TORCH_VERSION) >= digit_version('1.8'):
    #     bev_mask = torch.nan_to_num(bev_mask)
    # else:
    #     bev_mask = bev_mask.new_tensor(
    #         np.nan_to_num(bev_mask.cpu().numpy()))

    reference_points_cam = reference_points_cam.permute(1, 2, 0, 3)

    # print('bev_mask_mask:', bev_mask.shape) # (500, 500)
    # bev_mask = bev_mask.permute(1, 2, 0, 3).squeeze(-1)

    # print('reference_points_cam, bev_as_return:', reference_points_cam.shape)
    ### torch.Size([1, 32424, 1, 2])
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

        ## print('feat_feat1:', feat.size())
        # feat_feat1: torch.Size([1, 131072, 256])
        # feat_feat1: torch.Size([1, 32768, 256])
        # feat_feat1: torch.Size([1, 8192, 256])
        # feat_feat1: torch.Size([1, 2048, 256])


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

        # feat = feat.unsqueeze(1)
        # print('feathaha:', feat.size())
        feat = feat + level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
        # print('level_embeds:', level_embeds[None, None, lvl:lvl + 1, :])

        spatial_shapes.append(spatial_shape)
        feat_flatten.append(feat)
        # print('spatial_shapes_feat_flatten:', spatial_shapes, feat_flatten[0].size())
        ### [(116, 200), (58, 100), (29, 50), (15, 25)] spatial_shapes

    ###################################### plt feature map after backbone##########################################################

    # # print('feat_feat:', feat_flatten[2].size())
    # feat_show = feat_flatten[2]
    # feat_show = feat_show[0,:, :, 255]
    # # print('feat_feat2:', feat_show.size())
    # feat_show = feat_show.cpu().detach().numpy().astype(np.uint8)
    #
    # bev_embed_show = feat_show.reshape(64, 128, 1)
    # plt.imshow(bev_embed_show)
    # plt.title('Topdown semantic map prediction')
    # plt.axis('off')
    # plt.show()

    feat_flatten = torch.cat(feat_flatten, 2)
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=bev_pos.device)
    # print('feat_flatten:', feat_flatten.size())  ### torch.Size([6, 1, 174080, 256])

    level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
    # print('level_start_index_1:', level_start_index.size(), "value_value:", level_start_index)
    ### tensor([0, 23200, 29000, 30450]

    feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, H*W, bs, embed_dims) (6, 30825, 1, 256)

    return bev_queries, feat_flatten, bev_h, bev_w, bev_pos, spatial_shapes, level_start_index


if __name__ == '__main__':

    img_path = "/cvhci/data/segmentation/Stanford2D3D/matterport_pano/matterport3d3d/zsNo4HB9uLZ/pano_skybox_color/997b813443c64de5b38312642e937223.jpg"
    img = cv2.imread(img_path)
    img = cv2.resize(img, dsize=[2048, 1048])

    plt.imshow(img)
    plt.title('panoramic img')
    plt.axis('off')
    plt.show()


    gt_path = '/home/zteng/Desktop/haha/zsNo4HB9uLZ/997b813443c64de5b38312642e937223.h5'

    h5file = h5py.File(gt_path, 'r')
    map_heights = np.array(h5file['map_heights'])
    map_mask = np.array(h5file['mask'])
    gt_topdown = np.array(h5file['map_semantic'])
    # print('map_heights:', map_heights.shape, map_mask.shape, map_mask)

    # plt.imshow(img)
    plt.imshow(gt_topdown)
    plt.title('Topdown semantic map_gt')
    plt.axis('off')
    plt.show()


    H,W = 500,500
    ref_3d = get_reference_points(H, W, map_heights, map_mask, bs=1, device='cuda', dtype=torch.float)
    print('ref_3d:', ref_3d.shape)

    # import open3d as o3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(ref_3d)
    # o3d.visualization.draw_geometries([pcd])

    img_height, img_width = 1024, 2048
    # reference_points_cam = get_cam_reference_coordinate(ref_3d, img_height, img_width, img, map_mask)

    pc_range = [-5, -5, -1, 5, 5, 1]
    img_metas = [{'img_shape': [(1024, 2048, 3)]}]
    reference_points_cam, bev_mask = point_sampling_pano(ref_3d,  pc_range,  img_metas, map_mask)
    print('reference_points_cam_end:', reference_points_cam.shape, reference_points_cam.max())
