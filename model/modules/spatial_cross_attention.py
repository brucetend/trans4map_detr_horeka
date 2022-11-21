
# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------

from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init, constant_init
from mmcv.cnn.bricks.registry import (ATTENTION,
                                      TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
import math
from mmcv.runner import force_fp32, auto_fp16

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, MultiScaleDeformableAttnFunction_fp16
# from multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32, \
#     MultiScaleDeformableAttnFunction_fp16
# from projects.mmdet3d_plugin.models.utils.bricks import run_time
ext_module = ext_loader.load_ext(
    '_ext', ['ms_deform_attn_backward', 'ms_deform_attn_forward'])


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=1,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 # **kwargs2
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg  ## None
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims  # 256
        self.num_cams = num_cams      # 6
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.init_weight()
        # print('init_weight:', self.init_weight()) ### None

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                # **kwargs2
                ):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes. #### 为何reference points最后维度为4???

            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)

        if query_pos is not None:
            query = query + query_pos
        # print('query_pos11:', query_pos)   ### 测试中也为None,是因为temporal_self_attention加过了


        bs, num_query, _ = query.size()

        D = reference_points_cam.size(2)
        ## print('D:', D) # D == 4 pillar的数目
        indexes = []


        for i, mask_per_img in enumerate(bev_mask):
            # print('mask_per_img:', mask_per_img.size(), bev_mask[0].size())
            #### torch.Size([40000, 4])
            index_query_per_img = mask_per_img.sum(-1).nonzero().squeeze(-1) ### nonzero就是找到所有不为0的索引
            # print('index_query_per_img:', index_query_per_img.size(), index_query_per_img) ### 40000 顺序排列
            ### index_query_per_img: torch.Size([36638]) ###相当于40000格子中有的点和木有的点的index

            ## aaa = mask_per_img[0].sum(-1)
            indexes.append(index_query_per_img)
            ### print('max_len, indexes:', len(indexes)) ##### indexes = 1
            ### torch.Size([1, 40000, 4]) torch.Size([6301])

        max_len = max([len(each) for each in indexes])

        # each camera only interacts with its corresponding BEV queries. This step can greatly save GPU memory.
        queries_rebatch = query.new_zeros([bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros([bs, self.num_cams, max_len, D, 2])
        # print('queries_rebatch, reference_points_rebatch:', queries_rebatch.size(), reference_points_rebatch.size(), reference_points_cam.size())
        ### torch.Size([1, 6, 9675, 256]), torch.Size([1, 6, 9675, 4, 2]), torch.Size([1, 40000, 4, 2])

        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):

                reference_points_per_img = reference_points_per_img.unsqueeze(0)
                index_query_per_img = indexes[i]

                # print('reference_points_per_img, index_query_per_img:', reference_points_per_img.size(), index_query_per_img.size())
                ### torch.Size([40000, 4, 2]), torch.Size([36638])
                queries_rebatch[j, i,:len(index_query_per_img)] = query[j, index_query_per_img]

                reference_points_rebatch[j, i,:len(index_query_per_img)] = reference_points_per_img[j,index_query_per_img]
                device = 'cuda'
                reference_points_rebatch = reference_points_rebatch.to(device)
                # print('reference_points_rebatch:', reference_points_rebatch.size())
                #### 被mask出来的query和reference_points并且被裁剪 ### torch.Size([1, 1, 36638, 4, 2])

        ### print('queries_rebatch:', queries_rebatch.size())
        #### torch.Size([1, 6, 9675, 256])

        num_cams, l, bs, embed_dims = key.shape
        # print('key_shape:', key.shape)   ### torch.Size([6, 30825, 1, 256])

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        # print("key_value:", key.size(), value.size(), query.size())
        # torch.Size([6, 30825, 256]) torch.Size([6, 30825, 256]) torch.Size([1, 40000, 256])
        # print('level_start_index_0:', level_start_index)

        ## print('queries_in_deformable_attention:', key.size())


        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        # print('queries_after_attention:', queries.size()) ### MSDeformableAttention3D forward
        ### torch.Size([1, 1, 36638, 256])

        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]
        # print('slots:', slots.size(), queries[j, i, :len(index_query_per_img)].size())
        ### torch.Size([1, 40000, 256]) torch.Size([36638, 256])
        ### slots和query同格式，就是要把所有的queries都加上去，尽管短了很多!!!
        ### 不管原来有多短，都给他固定为40000长了~

        ## print('bev_mask_size:', bev_mask.size()) ### torch.Size([1, 40000, 4])
        bev_mask = bev_mask.to(device)
        count = bev_mask.sum(-1) > 0

        ### [1, 40000]
        ## count = count.permute(1, 2, 0).sum(-1)
        # print('count:', count.size(), count)

        ### 无论多小，最小值都是1
        count = torch.clamp(count, min=1.0)
        ### [1, 40000, 1]

        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4. ### 层数
        num_points (int): The number of sampling points for
            each query in each head. Default: 4. ### deformable_DETR里面的点
        im2col_step (int): The step used in image_to_column.
            Default: 64. ###
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8, #8
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):

        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step  ### 64

        self.embed_dims = embed_dims
        self.num_levels = num_levels # 4
        self.num_heads = num_heads   # 8
        self.num_points = num_points # 8

        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)

        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)

        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                # **kwarg
                 ):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        # print('bs_num_query_value:', bs, num_query, num_value, query.size())
        ### 6, 9675, 30258; bs = bs * num_cam
        # print('spatial_shapes:', spatial_shapes)
        # spatial_shapes: tensor([[116, 200],
        #                         [58, 100],
        #                         [29, 50],
        #                         [15, 25]], device='cuda:0')

        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        ## print('value_after_proj:', value.size()) ### torch.Size([6, 30825, 256])


        if key_padding_mask is not None:  #### key_padding_mask None
            value = value.masked_fill(key_padding_mask[..., None], 0.0)

        value = value.view(bs, num_value, self.num_heads, -1)

        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        # print('sampling_offsets_haha:', sampling_offsets.size())
        ## aaa - torch.Size([6, 9675, 512]), torch.Size([6, 9675, 8, 4, 8, 2]) 相当于query经过线性层 512 输出，拆成8,4,8,2


        # print('attention_weights:', query.size(), query.dtype)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)
        ### 线性层 self.attention_weights, torch.Size([6, 9675, 8, 32])

        #  print('attention_weights_0:', attention_weights[2, 2500:2600, :, :])  ###Attention的weight很奇怪
        attention_weights = attention_weights.softmax(-1)
        # print('attention_weights_1:', attention_weights[2, 2500:2600, :, :])

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)
        # print('attention_weights:', attention_weights[0, 1000, 2, :, :]) ### a lot if nan

        # print('args in attention:',self.num_heads, self.num_levels, self.num_points)
        ### 8 4 8


        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)
            ### calculate from spatial_shapes 重新写不同的normalizer给新的尺寸

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            ## torch.Size([1, 36638, 4, 2])

            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            # print('offset_normalizer:', sampling_offsets.size(), offset_normalizer) ### torch.Size([4, 2]), torch.Size([1, 36638, 8, 4, 8, 2])
                # tensor([[200, 116], Normalizer
                #         [100,  58],
                #         [ 50,  29],
                #         [ 25,  15]], device='cuda:0')

            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)

            # print('reference_points:', reference_points.size(), sampling_offsets.size(), reference_points.max(), reference_points.min(), sampling_offsets.max(), sampling_offsets.min())
            ### torch.Size([6, 36638, 1, 1, 1, 4, 2]) torch.Size([6, 36638, 8, 4, 2, 4, 2])

            sampling_locations = reference_points + sampling_offsets


            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors
            # print('sampling_offsets, sampling_locations:', sampling_offsets.size(), num_points * num_Z_anchors, num_points)
            ### sampling_offsets, sampling_locations: torch.Size([6, 9675, 8, 4, 2, 4, 2]) 8 2
            ### bs, num_query, num_heads-8, num_levels-4, num_points-2, num_Z_anchors-4

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #
        # print('cuda_device:', torch.cuda.is_available(), value.size(), value.is_cuda)
        ### cuda_device: True torch.Size([6, 30825, 8, 32]) True

        # print('level_start_index_1:', level_start_index) ###  tensor([0, 23200, 29000, 30450], device='cuda:0')


        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
            # print('MultiScaleDeformableAttnFunction:', value.size(), sampling_locations.size(), attention_weights.size(), output.size())
            ### torch.Size([6, 30825, 8, 32]) torch.Size([6, 9675, 8, 4, 8, 2]) torch.Size([6, 9675, 8, 4, 8]), torch.Size([1, 9675, 256])
            ### 8头注意力拆分！

        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)
        ## print('output_in_MSDeformableAttention3D:', output[2, 1300:1600,:])
        #### torch.Size([6, 9676, 256])

        return output

