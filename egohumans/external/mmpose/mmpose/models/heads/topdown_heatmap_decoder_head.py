# # Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import (build_conv_layer, build_norm_layer, build_upsample_layer,
                      constant_init, normal_init)
import warnings
from mmpose.core.evaluation import pose_pck_accuracy
from mmpose.core.post_processing import flip_back
from mmpose.models.builder import build_loss
from mmpose.models.utils.ops import resize
from ..builder import HEADS
import torch.nn.functional as F
from .topdown_heatmap_base_head import TopdownHeatmapBaseHead
from timm.models.layers import to_2tuple, trunc_normal_
from ..backbones.eva import SwiGLU, LayerNorm
import math
    
###---------------------------------------------------------
class MultiheadAttention(nn.Module):
    def __init__(self,
                    embed_dim=1024,
                    num_heads=16,
                    attn_drop=0.,
                    proj_drop=0.,
                    ):
        super().__init__()

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=attn_drop,)
        self.proj_drop = nn.Dropout(proj_drop)
        self.batch_first = True

        return
    
    def forward(self, query, key=None, value=None, identity=None, query_pos=None, key_pos=None, attn_mask=None, key_padding_mask=None):
        if key is None:
            key = query
        if value is None:
            value = key
        if identity is None:
            identity = query
        if key_pos is None:
            if query_pos is not None:
                # use query_pos if key_pos is not available
                if query_pos.shape == key.shape:
                    key_pos = query_pos
                else:
                    warnings.warn(f'position encoding of key is'
                                  f'missing in {self.__class__.__name__}.')
        if query_pos is not None:
            query = query + query_pos
        if key_pos is not None:
            key = key + key_pos

        if self.batch_first:
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)

        out = self.attn(query, key, value, attn_mask=attn_mask, key_padding_mask=key_padding_mask)[0] ## batch first is true

        if self.batch_first:
            out = out.transpose(0, 1)

        out = identity + self.proj_drop(out)

        return out

class DecoderBlock(nn.Module):
    def __init__(self,
                    embed_dim=1024,
                    num_heads=16,
                    dropout=0.,
                    mlp_ratio=2.6666666666666665,
                    norm_layer=nn.LayerNorm,
                    ):
        super().__init__()

        self.self_attn = MultiheadAttention(embed_dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        self.cross_attn = MultiheadAttention(embed_dim, num_heads, attn_drop=dropout, proj_drop=dropout)
        
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.norm3 = norm_layer(embed_dim)

        hidden_features = int(embed_dim * mlp_ratio)
        self.mlp = SwiGLU(
                    in_features=embed_dim,
                    hidden_features=hidden_features,
                    norm_layer=LayerNorm,
                    drop=dropout,
                )

        return
    
    def forward(self, query, key=None, value=None, query_pos=None, key_pos=None, self_attn_mask=None, cross_attn_mask=None, key_padding_mask=None):

        query = self.self_attn(query, key=query, value=query, query_pos=query_pos, key_pos=query_pos, attn_mask=self_attn_mask)
        query = query + self.norm1(query)

        query = self.cross_attn(query, key=key, value=value, query_pos=query_pos, key_pos=key_pos, attn_mask=cross_attn_mask, key_padding_mask=key_padding_mask)
        query = query + self.mlp(self.norm2(query)) 

        query = self.norm3(query)

        return query


@HEADS.register_module()
class TopdownHeatmapDecoderHead(TopdownHeatmapBaseHead):
    def __init__(self,
                 encoder_embed_dim=1024,
                 decoder_embed_dim=256,
                 num_joints=17,
                 num_deconv_layers=3,
                 num_deconv_filters=(256, 256, 256),
                 num_deconv_kernels=(4, 4, 4),
                 extra=None,
                 num_layers=6,
                 num_heads=16,
                 dropout=0.,
                 num_patch_h=32,
                 num_patch_w=24,
                 in_index=0,
                 input_transform=None,
                 align_corners=False,
                 loss_keypoint=None,
                 train_cfg=None,
                 test_cfg=None,
                 upsample=0,):
        super().__init__()

        self.encoder_embed_dim = encoder_embed_dim
        self.decoder_embed_dim = decoder_embed_dim

        ## if using transformer
        in_channels = encoder_embed_dim
        out_channels = decoder_embed_dim

        ## for debug
        # in_channels = encoder_embed_dim
        # out_channels = 17

        self.loss = build_loss(loss_keypoint)
        self.upsample = upsample

        self.train_cfg = {} if train_cfg is None else train_cfg
        self.test_cfg = {} if test_cfg is None else test_cfg
        self.target_type = self.test_cfg.get('target_type', 'GaussianHeatmap')

        self._init_inputs(in_channels, in_index, input_transform)
        self.in_index = in_index
        self.align_corners = align_corners

        if extra is not None and not isinstance(extra, dict):
            raise TypeError('extra should be dict or None.')

        if num_deconv_layers > 0:
            self.deconv_layers = self._make_deconv_layer(
                num_deconv_layers,
                num_deconv_filters,
                num_deconv_kernels,
            )
        elif num_deconv_layers == 0:
            self.deconv_layers = nn.Identity()
        else:
            raise ValueError(
                f'num_deconv_layers ({num_deconv_layers}) should >= 0.')

        identity_final_layer = False
        if extra is not None and 'final_conv_kernel' in extra:
            assert extra['final_conv_kernel'] in [0, 1, 3]
            if extra['final_conv_kernel'] == 3:
                padding = 1
            elif extra['final_conv_kernel'] == 1:
                padding = 0
            else:
                # 0 for Identity mapping.
                identity_final_layer = True
            kernel_size = extra['final_conv_kernel']
        else:
            kernel_size = 1
            padding = 0

        if identity_final_layer:
            self.final_layer = nn.Identity()
        else:
            conv_channels = num_deconv_filters[
                -1] if num_deconv_layers > 0 else self.in_channels

            layers = []
            if extra is not None:
                num_conv_layers = extra.get('num_conv_layers', 0)
                num_conv_kernels = extra.get('num_conv_kernels',
                                             [1] * num_conv_layers)

                for i in range(num_conv_layers):
                    layers.append(
                        build_conv_layer(
                            dict(type='Conv2d'),
                            in_channels=conv_channels,
                            out_channels=conv_channels,
                            kernel_size=num_conv_kernels[i],
                            stride=1,
                            padding=(num_conv_kernels[i] - 1) // 2))
                    layers.append(
                        build_norm_layer(dict(type='BN'), conv_channels)[1])
                    layers.append(nn.ReLU(inplace=True))

            layers.append(
                build_conv_layer(
                    cfg=dict(type='Conv2d'),
                    in_channels=conv_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    padding=padding))

            if len(layers) > 1:
                self.final_layer = nn.Sequential(*layers)
            else:
                self.final_layer = layers[0]

        ## Decoder layers
        # self.keypoint_query_embed = nn.Parameter(torch.zeros(1, num_joints, decoder_embed_dim)) ## 1, 17, 256
        # trunc_normal_(self.keypoint_query_embed, std=.02)
        
        self.keypoint_pos_embed = nn.Embedding(num_joints, decoder_embed_dim) ## 17, 256

        self.encoder_patch_pos_embed = nn.Parameter(torch.zeros(1, num_patch_h*num_patch_w, encoder_embed_dim)) ## 1, 768 (32 x 24), 1024
        trunc_normal_(self.encoder_patch_pos_embed, std=.02)        

        self.decoder_patch_pos_embed = nn.Parameter(torch.zeros(1, num_patch_h*num_patch_w, decoder_embed_dim)) ## 1, 768 (32 x 24), 256
        trunc_normal_(self.decoder_patch_pos_embed, std=.02)

        self.decoder_input_proj = nn.Conv2d(encoder_embed_dim, decoder_embed_dim, kernel_size=1) ## 1024, 256, 1

        self.blocks = nn.ModuleList([
                        DecoderBlock(embed_dim=decoder_embed_dim, num_heads=num_heads, dropout=dropout, norm_layer=nn.LayerNorm)
                        for _ in range(num_layers)])
        
        self.norm = nn.LayerNorm(decoder_embed_dim) ## 256

        self.keypoint_embed_mlp = nn.Sequential(
                                nn.Linear(decoder_embed_dim, decoder_embed_dim), nn.ReLU(inplace=True),
                                nn.Linear(decoder_embed_dim, decoder_embed_dim), nn.ReLU(inplace=True),
                                nn.Linear(decoder_embed_dim, decoder_embed_dim), 
                            )

        return

    def get_loss(self, output, target, target_weight, factor=10.0):
        """Calculate top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        losses = dict()

        assert not isinstance(self.loss, nn.Sequential)
        assert target.dim() == 4 and target_weight.dim() == 3

        if factor != 1.0:
            losses['heatmap_loss'] = factor*self.loss(output, target, target_weight)

        return losses

    def get_accuracy(self, output, target, target_weight):
        """Calculate accuracy for top-down keypoint loss.

        Note:
            - batch_size: N
            - num_keypoints: K
            - heatmaps height: H
            - heatmaps weight: W

        Args:
            output (torch.Tensor[N,K,H,W]): Output heatmaps.
            target (torch.Tensor[N,K,H,W]): Target heatmaps.
            target_weight (torch.Tensor[N,K,1]):
                Weights across different joint types.
        """

        accuracy = dict()

        if self.target_type == 'GaussianHeatmap':
            _, avg_acc, _ = pose_pck_accuracy(
                output.detach().cpu().numpy(),
                target.detach().cpu().numpy(),
                target_weight.detach().cpu().numpy().squeeze(-1) > 0)
            accuracy['acc_pose'] = float(avg_acc)

        return accuracy

    def forward(self, x):
        """Forward function."""
        x = self._transform_inputs(x)

        B, C, Hp, Wp = x.shape

        x_decoder = self.decoder_input_proj(x) ## B x 1024 x 32 x 24 -> B x 256 x 32 x 24

        # shape = (batch_size, num_queries, embed_dims)
        # query_embed = self.keypoint_query_embed.repeat(B, 1, 1) ## B x 17 x 256
        query_embed = self.keypoint_pos_embed.weight.unsqueeze(0).repeat(B, 1, 1) ## B x 17 x 256

        ## reshape embedding and repeat
        encoder_patch_pos_embed = self.encoder_patch_pos_embed.reshape(1, -1, Hp, Wp) ## 1 x 1024 x 32 x 24
        encoder_patch_pos_embed = encoder_patch_pos_embed.repeat(B, 1, 1, 1) ## B x 1024 x 32 x 24

        decoder_patch_pos_embed = self.decoder_patch_pos_embed.repeat(B, 1, 1) ## B x 768 x 256

        ## flatten
        x_token = x_decoder.permute(0, 2, 3, 1).reshape(B, Hp*Wp, -1) ## B x 768 x 256

        ## define variable
        query = torch.zeros_like(query_embed)
        key = x_token
        value = x_token

        for blk in self.blocks:
            query = blk(query, key=key, value=value, query_pos=query_embed, key_pos=decoder_patch_pos_embed)

        out_decoder = self.norm(query) ## B x 17 x 256
        keypoint_embed = self.keypoint_embed_mlp(out_decoder) ## B x 17 x 256

        ## extract keypoint image features from x, gradually upsample
        x = self.deconv_layers(x + encoder_patch_pos_embed) ## B x 1024 x 128 x 96
        x = self.final_layer(x) ## B x 256 x 128 x 96

        heatmap = torch.einsum('bqc,bchw->bqhw', keypoint_embed, x) ## B x 17 x 128 x 96

        return heatmap

    def inference_model(self, x, flip_pairs=None):
        """Inference function.

        Returns:
            output_heatmap (np.ndarray): Output heatmaps.

        Args:
            x (torch.Tensor[N,K,H,W]): Input features.
            flip_pairs (None | list[tuple]):
                Pairs of keypoints which are mirrored.
        """
        output = self.forward(x)

        if flip_pairs is not None:
            output_heatmap = flip_back(
                output.detach().cpu().numpy(),
                flip_pairs,
                target_type=self.target_type)
            # feature is not aligned, shift flipped heatmap for higher accuracy
            if self.test_cfg.get('shift_heatmap', False):
                output_heatmap[:, :, :, 1:] = output_heatmap[:, :, :, :-1]
        else:
            output_heatmap = output.detach().cpu().numpy()
        return output_heatmap

    def _init_inputs(self, in_channels, in_index, input_transform):
        """Check and initialize input transforms.

        The in_channels, in_index and input_transform must match.
        Specifically, when input_transform is None, only single feature map
        will be selected. So in_channels and in_index must be of type int.
        When input_transform is not None, in_channels and in_index must be
        list or tuple, with the same length.

        Args:
            in_channels (int|Sequence[int]): Input channels.
            in_index (int|Sequence[int]): Input feature index.
            input_transform (str|None): Transformation type of input features.
                Options: 'resize_concat', 'multiple_select', None.

                - 'resize_concat': Multiple feature maps will be resize to the
                    same size as first one and than concat together.
                    Usually used in FCN head of HRNet.
                - 'multiple_select': Multiple feature maps will be bundle into
                    a list and passed into decode head.
                - None: Only one select feature map is allowed.
        """

        if input_transform is not None:
            assert input_transform in ['resize_concat', 'multiple_select']
        self.input_transform = input_transform
        self.in_index = in_index
        if input_transform is not None:
            assert isinstance(in_channels, (list, tuple))
            assert isinstance(in_index, (list, tuple))
            assert len(in_channels) == len(in_index)
            if input_transform == 'resize_concat':
                self.in_channels = sum(in_channels)
            else:
                self.in_channels = in_channels
        else:
            assert isinstance(in_channels, int)
            assert isinstance(in_index, int)
            self.in_channels = in_channels

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.

        Args:
            inputs (list[Tensor] | Tensor): multi-level img features.

        Returns:
            Tensor: The transformed inputs
        """
        if not isinstance(inputs, list):
            if not isinstance(inputs, list):
                if self.upsample > 0:
                    inputs = resize(
                        input=F.relu(inputs),
                        scale_factor=self.upsample,
                        mode='bilinear',
                        align_corners=self.align_corners
                        )
            return inputs

        if self.input_transform == 'resize_concat':
            inputs = [inputs[i] for i in self.in_index]
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners) for x in inputs
            ]
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == 'multiple_select':
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def _make_deconv_layer(self, num_layers, num_filters, num_kernels):
        """Make deconv layers."""
        if num_layers != len(num_filters):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_filters({len(num_filters)})'
            raise ValueError(error_msg)
        if num_layers != len(num_kernels):
            error_msg = f'num_layers({num_layers}) ' \
                        f'!= length of num_kernels({len(num_kernels)})'
            raise ValueError(error_msg)

        layers = []
        for i in range(num_layers):
            kernel, padding, output_padding = \
                self._get_deconv_cfg(num_kernels[i])

            planes = num_filters[i]
            layers.append(
                build_upsample_layer(
                    dict(type='deconv'),
                    in_channels=self.in_channels,
                    out_channels=planes,
                    kernel_size=kernel,
                    stride=2,
                    padding=padding,
                    output_padding=output_padding,
                    bias=False))
            layers.append(nn.BatchNorm2d(planes))
            layers.append(nn.ReLU(inplace=True))
            self.in_channels = planes

        return nn.Sequential(*layers)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.self_attn.attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.cross_attn.attn.out_proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        return

    def init_weights(self):
        """Initialize model weights."""

        # init weights
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)

        self.apply(_init_weights)
        self.fix_init_weight()

        for _, m in self.deconv_layers.named_modules():
            if isinstance(m, nn.ConvTranspose2d):
                normal_init(m, std=0.001)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
        for m in self.final_layer.modules():
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.001, bias=0)
            elif isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
