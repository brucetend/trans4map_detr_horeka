
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

#from projects.mmdet3d_plugin.models.utils.bricks import run_time
#from projects.mmdet3d_plugin.models.utils.visual import save_tensor
from model.modules.point_sampling_panorama import point_sampling_pano
from .custom_base_transformer_layer import MyCustomBaseTransformerLayer

import matplotlib.pyplot as plt
import copy
import warnings
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)

from mmcv.cnn.bricks.transformer import TransformerLayerSequence
from mmcv.runner import force_fp32, auto_fp16
import numpy as np
import torch
import cv2 as cv
import mmcv
from mmcv.utils import TORCH_VERSION, digit_version
from mmcv.utils import ext_loader
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@TRANSFORMER_LAYER_SEQUENCE.register_module()
class BEVFormerEncoder(TransformerLayerSequence):
    #### BEVFormerEncoder 包含 BEVFormerLayer,继承类来自文件transformer.py
    #### 主要任务产生reference points 和 query
    """
    Attention with both self and cross
    Implements the de(en)coder in DETR transformer.
    Args:
        return_intermediate (bool): Whether to return intermediate outputs.
        coder_norm_cfg (dict): Config of last normalization layer. Default：
            `LN`.
    """

    def __init__(self, *args, pc_range=None, num_points_in_pillar=4, return_intermediate=False, dataset_type='nuscenes',
                 **kwargs):

        super(BEVFormerEncoder, self).__init__(*args, **kwargs)
        self.return_intermediate = return_intermediate

        self.num_points_in_pillar = num_points_in_pillar
        self.pc_range = pc_range
        self.fp16_enabled = False
        # print('kwargs in BEVFormerEncoder:', kwargs.keys())

    # @staticmethod
    # def get_reference_points_pano(H, W, Z=8, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
    #     """Get the reference points used in SCA and TSA. 获得reference points
    #         Args:
    #         H, W: spatial shape of bev.  H=W=200
    #         Z: hight of pillar.
    #         D: sample D points uniformly from each pillar. num_points_in_pillar = 4
    #         device (obj:`device`): The device where reference_points should be.
    #     Returns:
    #         Tensor: reference points used in decoder, has \
    #             shape (bs, num_keys, num_levels, 2).
    #     """

    @staticmethod
    def get_reference_points(H, W, map_heights, map_mask, bs=1, device='cuda', dtype=torch.float):

        row_column_index = torch.where(map_mask == True)
        row = row_column_index[0]
        column = row_column_index[1]

        x_pos = row * 0.02 - 0.01 - 5
        y_pos = (H - column) * 0.02 + 0.01 - 5
        z_pos = map_heights[0, row, column] - 10.0
        # print('position_haha:', row.size(), z_pos.size(), map_heights.size())

        real_position = torch.stack((x_pos, y_pos, z_pos), axis=1)

        ref_3d = real_position.unsqueeze(0)
        ref_3d = ref_3d.unsqueeze(0)
        # print('real_position:', ref_3d.shape, real_position[:, 2].min(), real_position[:, 2].max())
        # (1, 1, 32424, 3)
        return ref_3d

        # def get_reference_points(H, W, Z=2, num_points_in_pillar=4, dim='3d', bs=1, device='cuda', dtype=torch.float):
    #     """Get the reference points used in SCA and TSA. 获得reference points
    #     Args:
    #         H, W: spatial shape of bev.
    #         Z: hight of pillar.
    #         D: sample D points uniformly from each pillar.
    #         device (obj:`device`): The device where
    #             reference_points should be.
    #     Returns:
    #         Tensor: reference points used in decoder, has \
    #             2D: (bs, num_keys, num_levels, 2).
    #             3D:  [1, 4, 40000, 3]
    #     """
    #
    #     # reference points in 3D space, used in spatial cross-attention (SCA)
    #     if dim == '3d':
    #         zs = torch.linspace(0.5, Z - 0.5, num_points_in_pillar, dtype=dtype,
    #                             device=device).view(-1, 1, 1).expand(num_points_in_pillar, H, W) / Z
    #         xs = torch.linspace(0.5, W - 0.5, W, dtype=dtype,
    #                             device=device).view(1, 1, W).expand(num_points_in_pillar, H, W) / W
    #         ys = torch.linspace(0.5, H - 0.5, H, dtype=dtype,
    #                             device=device).view(1, H, 1).expand(num_points_in_pillar, H, W) / H
    #         ref_3d = torch.stack((xs, ys, zs), -1)
    #         # print('ref_3d_0:', ref_3d.size())
    #
    #         ref_3d = ref_3d.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
    #         ref_3d = ref_3d[None].repeat(bs, 1, 1, 1)
    #         # print('ref_3d:', ref_3d.size())  ### torch.Size # 3D:  [1, 4, 40000, 3]
    #
    #         #### 画图 可视化
    #
    #         # ref_3d_ = ref_3d.cpu()
    #         #
    #         # fig = plt.figure()
    #         # ax = fig.add_subplot(projection = '3d')
    #         # xss = ref_3d_[0, 0, :, 0]
    #         # yss = ref_3d_[0, 0, :, 1]
    #         # zss = ref_3d_[0, 0, :, 2]
    #         #
    #         # ax.scatter(xss, yss, zss)
    #         #
    #         # ax.set_xlabel('X Label')
    #         # ax.set_ylabel('Y Label')
    #         # ax.set_zlabel('Z Label')
    #         # plt.show()
    #
    #         return ref_3d


        # # reference points on 2D bev plane, used in temporal self-attention (TSA).
        # elif dim == '2d':
        #     ref_y, ref_x = torch.meshgrid(
        #         torch.linspace(
        #             0.5, H - 0.5, H, dtype=dtype, device=device),
        #         torch.linspace(
        #             0.5, W - 0.5, W, dtype=dtype, device=device)
        #     )
        #     ref_y = ref_y.reshape(-1)[None] / H
        #     ref_x = ref_x.reshape(-1)[None] / W
        #     ref_2d = torch.stack((ref_x, ref_y), -1)
        #     ref_2d = ref_2d.repeat(bs, 1, 1).unsqueeze(2)  ### torch.Size([2, 40000, 1, 2])
        #     # fig = plt.figure()
        #     #
        #     # ref_2d_ = ref_2d.cpu()
        #     # ax = fig.add_subplot(projection = '3d')
        #     # xss = ref_2d_[0, :, 0, 0]
        #     # yss = ref_2d_[0, :, 0, 1]
        #     # print('xss,yss:', xss.min(), xss.max(), len(xss),yss.min(), yss.max())
        #     #
        #     # ax.scatter(xss, yss)
        #     #
        #     # ax.set_xlabel('X Label')
        #     # ax.set_ylabel('Y Label')
        #     # ax.set_zlabel('Z Label')
        #     #
        #     # plt.show()
        #     return ref_2d

    # This function must use fp32!!!
    # @force_fp32(apply_to=('reference_points', 'img_metas')
    # def point_sampling(self, reference_points, pc_range,  img_metas):
    #     ##### reference point 和 pc_range 还有变换矩阵换进来, got reference_points_cam and bev_mask
    #     lidar2img = []
    #     for img_meta in img_metas:
    #         lidar2img.append(img_meta['lidar2img'])
    #         ## print("lidar2img_0:", len(lidar2img))
    #
    #     lidar2img = np.asarray(lidar2img)
    #     # print('lidar2img-1:', lidar2img.shape, lidar2img)
    #
    #     lidar2img = reference_points.new_tensor(lidar2img)  # (B, N, 4, 4) 只是为了延续reference_points tensor的dtype和device
    #     ## print('lidar2img1:', lidar2img.size(), lidar2img)
    #     ## torch.Size([1, 6, 4, 4])
    #     reference_points = reference_points.clone()
    #     # print("###***:", reference_points.size())
    #     ### torch.Size([1, 4, 40000, 3])
    #
    #
    #     reference_points[..., 0:1] = reference_points[..., 0:1] * \
    #         (pc_range[3] - pc_range[0]) + pc_range[0]
    #     reference_points[..., 1:2] = reference_points[..., 1:2] * \
    #         (pc_range[4] - pc_range[1]) + pc_range[1]
    #     reference_points[..., 2:3] = reference_points[..., 2:3] * \
    #         (pc_range[5] - pc_range[2]) + pc_range[2]
    #
    #     # print('reference_points_why:', reference_points.size())  ### torch.Size([1, 4, 40000, 3])
    #
    #     reference_points = torch.cat(
    #         (reference_points, torch.ones_like(reference_points[..., :1])), -1)
    #     # print('reference_point_0:', reference_points.size()) #### torch.Size([1, 4, 40000, 4]) ##相比之前只是加上了一层全是1的
    #
    #     reference_points = reference_points.permute(1, 0, 2, 3)
    #     # print('reference_point_1:', reference_points.size()) #### torch.Size([4, 1, 40000, 4])
    #
    #     D, B, num_query = reference_points.size()[:3]
    #     ### num_cam = lidar2img.size(1)     ## 6 cameras
    #     num_cam = 1
    #
    #     reference_points = reference_points.view(
    #         D, B, 1, num_query, 4).repeat(1, 1, num_cam, 1, 1).unsqueeze(-1)
    #     # print('reference_point_2:', reference_points.size())  ### torch.Size([4, 1, 6, 40000, 4, 1])
    #     ## torch.Size([4, 1, 1, 40000, 4, 1])
    #
    #     lidar2img = lidar2img[:, 1, :,:]
    #     lidar2img = lidar2img.view(1, B, num_cam, 1, 4, 4).repeat(D, 1, 1, num_query, 1, 1)
    #     # print('lidar2img2:', lidar2img.size())  ### torch.Size([4, 1, 6, 40000, 4, 4])
    #     #### torch.Size([4, 1, 1, 40000, 4, 4])
    #
    #
    #     reference_points_cam = torch.matmul(lidar2img.to(torch.float32), reference_points.to(torch.float32)).squeeze(-1)
    #     ### 矩阵乘法在这里有何作用，只有最后两个维度？坐标转换？把cam转化为...
    #     # reference_points_cam22 = torch.matmul(lidar2img.to(torch.float32),
    #     #                                     reference_points.to(torch.float32))
    #     ### print('reference_points_cam:', reference_points_cam[0,0,0,0,:])  ### torch.Size([4, 1, 6, 40000, 4])
    #     ### torch.Size([4, 1, 1, 40000, 4])
    #
    #     eps = 1e-5
    #     bev_mask = (reference_points_cam[..., 2:3] > eps)
    #     # print('bev_mask_0:', bev_mask.size())  ### torch.Size([4, 1, 6, 40000, 1])
    #     ### torch.Size([4, 1, 1, 40000, 1])
    #     ### 经过RT矩阵投影的，z坐标比0大的东西，camera坐标系下
    #     # print('reference_points_cam[..., 2:3]:', reference_points_cam[..., 2:3])
    #
    #     reference_points_cam = reference_points_cam[..., 0:2] / torch.maximum(
    #         reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3]) * eps)
    #     ### torch.Size([4, 1, 1, 40000, 2])
    #
    #     reference_points_cam[..., 0] /= img_metas[0]['img_shape'][0][1] # 1600
    #     reference_points_cam[..., 1] /= img_metas[0]['img_shape'][0][0] # 928
    #
    #     bev_mask = (bev_mask & (reference_points_cam[..., 1:2] > 0.0)
    #                 & (reference_points_cam[..., 1:2] < 1.0)
    #                 & (reference_points_cam[..., 0:1] < 1.0)
    #                 & (reference_points_cam[..., 0:1] > 0.0))
    #
    #     if digit_version(TORCH_VERSION) >= digit_version('1.8'):
    #         bev_mask = torch.nan_to_num(bev_mask)
    #     else:
    #         bev_mask = bev_mask.new_tensor(
    #             np.nan_to_num(bev_mask.cpu().numpy()))
    #
    #     reference_points_cam = reference_points_cam.permute(2, 1, 3, 0, 4)
    #     bev_mask = bev_mask.permute(2, 1, 3, 0, 4).squeeze(-1)
    #
    #     # print('reference_points_cam, bev_as_return:', reference_points_cam, bev_mask.size())
    #     ### torch.Size([6, 1, 40000, 4, 2]) torch.Size([6, 1, 40000, 4])
    #     ### torch.Size([1, 1, 40000, 4, 2]) torch.Size([1, 1, 40000, 4])
    #     return reference_points_cam, bev_mask

    @auto_fp16()
    def forward(self,
                bev_query,
                key,
                value,
                *args,
                bev_h=None,
                bev_w=None,
                bev_pos=None,
                spatial_shapes=None,
                level_start_index=None,
                valid_ratios=None,
                prev_bev=None,
                shift=0.,
                map_mask=None,
                map_heights=None,
                **kwargs):
        """Forward function for `TransformerDecoder`.
        Args:
            bev_query (Tensor): Input BEV query with shape
                `(num_query, bs, embed_dims)`.
            key & value (Tensor): Input multi-cameta features with shape
                (num_cam, num_value, bs, embed_dims)
            reference_points (Tensor): The reference
                points of offset. has shape
                (bs, num_query, 4) when as_two_stage, 所以“two_stage”是什么
                otherwise has shape ((bs, num_query, 2).
            valid_ratios (Tensor): The radios of valid
                points on the feature map, has shape
                (bs, num_levels, 2)
            map_heights, map_mask 用以辅助生成足够好的ref_3d
        Returns:
            Tensor: Results with shape [1, num_query, bs, embed_dims] when
                return_intermediate is `False`, otherwise it has shape
                [num_layers, num_query, bs, embed_dims].
        """
        # print("kwargs_in_BEVFormerEncoder_in_forward:", kwargs.keys()) ### dict_keys(['img_metas'])

        output = bev_query
        intermediate = []

        #################################################################################################################
        # ref_3d = self.get_reference_points(
        #     bev_h, bev_w, self.pc_range[5]-self.pc_range[2], self.num_points_in_pillar, dim='3d', bs=bev_query.size(1),  device=bev_query.device, dtype=bev_query.dtype) ### [1, 4, 40000, 3]
        # ref_2d = self.get_reference_points(
        #     bev_h, bev_w, dim='2d', bs=bev_query.size(1), device=bev_query.device, dtype=bev_query.dtype)   ### [1, 40000, 1, 2]

        ref_3d =  self.get_reference_points(bev_h, bev_w, map_heights, map_mask, bs=1, device='cuda', dtype=torch.float)

        # reference_points_cam, bev_mask = self.point_sampling(ref_3d, self.pc_range, kwargs['img_metas'])
        reference_points_cam, bev_mask = point_sampling_pano(ref_3d, self.pc_range, kwargs['img_metas'], map_mask)

        # print('reference_points_cam:', reference_points_cam.size(), bev_mask.size())
        #### 用reference point算出了reference_points_cam, bev_mask，结合pc_range, XYZ方向上的范围
        ### torch.Size([1, 40000, 4, 2]) torch.Size([1, 40000, 4])
        ################################################################################################################


        # bug: this code should be 'shift_ref_2d = ref_2d.clone()', we keep this bug for reproducing our results in paper.
        # shift_ref_2d = ref_2d  # .clone()
        # shift_ref_2d += shift[:, None, None, :]

        # (num_query, bs, embed_dims) -> (bs, num_query, embed_dims)
        bev_query = bev_query.permute(1, 0, 2)
        bev_pos = bev_pos.permute(1, 0, 2)
        # print("bev_pos:", bev_pos.size())   ### torch.Size([1, 40000, 256])

        # bs, len_bev, num_bev_level, _ = ref_2d.shape
        # print('ref2d_bs：', bs, len_bev, num_bev_level)   ### torch.Size([1, 40000,1, 2])
        # print('prev_bev:', prev_bev) ## None


        for lid, layer in enumerate(self.layers):
            # print('layer:', layer) ### BEVFormerLayer as forward

            output = layer(
                bev_query,
                key,
                value,
                *args,
                bev_pos=bev_pos,
                # ref_2d=hybird_ref_2d,
                ref_3d=ref_3d,
                bev_h=bev_h,
                bev_w=bev_w,
                spatial_shapes=spatial_shapes,
                level_start_index=level_start_index,
                bev_mask=bev_mask,
                reference_points_cam=reference_points_cam,
                prev_bev=prev_bev,
                **kwargs
                )

            bev_query = output
            if self.return_intermediate:
                intermediate.append(output)

        # print('self.layer:', layer, "intermediate:", intermediate)  ## 在for循环中运行6次 intermediate = None

        if self.return_intermediate:
            return torch.stack(intermediate)
        # print("output in Encoder:", output.size())  ### torch.Size([1, 40000, 256])
        return output


@TRANSFORMER_LAYER.register_module()
class BEVFormerLayer(MyCustomBaseTransformerLayer):
    """Implements decoder layer in DETR transformer.
    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | list[dict] | dict )):
            Configs for self_attention or cross_attention, the order
            should be consistent with it in `operation_order`. If it is
            a dict, it would be expand to the number of attention in
            `operation_order`.
        feedforward_channels (int): The hidden dimension for FFNs.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0.0.
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Default：None
        act_cfg (dict): The activation config for FFNs. Default: `LN`
        norm_cfg (dict): Config dict for normalization layer.
            Default: `LN`.
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
    """

    def __init__(self,
                 attn_cfgs,
                 feedforward_channels,
                 ffn_dropout=0.0,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 **kwargs):
        super(BEVFormerLayer, self).__init__(
            attn_cfgs=attn_cfgs,
            feedforward_channels=feedforward_channels,
            ffn_dropout=ffn_dropout,
            operation_order=operation_order,
            act_cfg=act_cfg,
            norm_cfg=norm_cfg,
            ffn_num_fcs=ffn_num_fcs,
            **kwargs)
        self.fp16_enabled = False
        # assert len(operation_order) == 6
        assert len(operation_order) == 4
        # assert set(operation_order) == set(['self_attn', 'norm', 'cross_attn', 'ffn'])
        assert set(operation_order) == set(['norm', 'cross_attn', 'ffn'])
        ### print('kwargs_in BEVFormerLayer:', kwargs.keys())    ## dict_keys('img_metas')


    def forward(self,
                query,
                key=None,
                value=None,
                bev_pos=None,
                # query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                # ref_2d=None,
                ref_3d=None,
                bev_h=None,
                bev_w=None,
                reference_points_cam=None,
                bev_mask=None,
                spatial_shapes=None,
                level_start_index=None,
                prev_bev=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        ### print('query_pos:', query_pos, 'key_pos:', key_pos)
        ### 至少在 BEVFormerLayer 里，query_pos和 key_pos并没有被输入 None

        # print('kwargs_in BEVFormerLayer_forward:', kwargs.keys())

        norm_index = 0
        attn_index = 0
        ffn_index = 0

        # print('bev_pos_in_BEVFormerLayer:', bev_pos.size())

        identity = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                                                     f'attn_masks {len(attn_masks)} must be equal ' \
                                                     f'to the number of attention in ' \
                f'operation_order {self.num_attn}'

        # print('operation_order:', self.operation_order)
        ### operation_order: ('cross_attn', 'norm', 'ffn', 'norm')

        for layer in self.operation_order:
            # temporal self attention
            if layer == 'self_attn':

                print("self_attn_index:", self.attentions[attn_index])

                # query = self.attentions[attn_index](
                #     query,
                #     prev_bev,
                #     prev_bev,
                #     identity if self.pre_norm else None,
                #     query_pos=bev_pos,
                #     key_pos=bev_pos,
                #     attn_mask=attn_masks[attn_index],
                #     key_padding_mask=query_key_padding_mask,
                #     reference_points=ref_2d,
                #     spatial_shapes=torch.tensor(
                #         [[bev_h, bev_w]], device=query.device),
                #     level_start_index=torch.tensor([0], device=query.device),
                #     **kwargs)
                # attn_index += 1
                # identity = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                # print('query_in_norm:', query.size())
                norm_index += 1

            # spaital cross attention
            elif layer == 'cross_attn':
                # print("cross_attn_index:", self.attentions[attn_index])
                # print('query_pos:', bev_pos.size())

                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    identity if self.pre_norm else None,
                    # query_pos=query_pos,
                    query_pos=bev_pos,
                    # key_pos=key_pos,
                    reference_points=ref_3d,
                    reference_points_cam=reference_points_cam,
                    bev_mask= bev_mask,
                    # attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    # **kwargs2
                )
                # print('after_cross_attention:', query.size())

                attn_index += 1
                identity = query

            elif layer == 'ffn':
                # print("ffn_index:", ffn_index)  ### ffn_index: 0

                query = self.ffns[ffn_index](
                    query, identity if self.pre_norm else None)

                # print('query_in_ffn:', query.size())
                ffn_index += 1

        return query

