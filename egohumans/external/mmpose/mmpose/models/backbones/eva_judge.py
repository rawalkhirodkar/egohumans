# Copyright (c) OpenMMLab. All rights reserved.
import math
from typing import Callable, Optional, Tuple, Union, List
import torch
from functools import partial
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from timm.models.layers import to_2tuple, trunc_normal_

from ..builder import BACKBONES
from .base_backbone import BaseBackbone
from .eva import *

@BACKBONES.register_module()
class EvaJudge(BaseBackbone):

    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            num_classes: int = 1000,
            embed_dim: int = 768,
            depth: int = 12,
            depth_heatmap_idxs=[6, 8, 10, 12], ## default is depth 24
            num_heads: int = 12,
            qkv_bias: bool = True,
            qkv_fused: bool = True,
            mlp_ratio: float = 4.,
            swiglu_mlp: bool = False,
            scale_mlp: bool = False,
            scale_attn_inner: bool = False,
            drop_rate: float = 0.,
            pos_drop_rate: float = 0.,
            patch_drop_rate: float = 0.,
            proj_drop_rate: float = 0.,
            attn_drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            norm_layer: Callable = LayerNorm,
            init_values: Optional[float] = None,
            use_eva02: bool = False, ## orginailly class_token
            use_rot_pos_emb: bool = False,
            use_post_norm: bool = False,
            last_norm: bool = True,
            ref_feat_shape: Optional[Union[Tuple[int, int], int]] = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.patch_embed_heatmap = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=1,
            embed_dim=embed_dim,
        )

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_eva02 else None
        self.cls_token_heatmap = nn.Parameter(torch.zeros(1, 1, embed_dim)) if use_eva02 else None

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) ## because of class token
        self.pos_drop = nn.Dropout(p=pos_drop_rate)

        self.pos_embed_heatmap = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) ## because of class token

        if patch_drop_rate > 0:
            self.patch_drop = PatchDropout(
                patch_drop_rate,
                num_prefix_tokens=1,
                return_indices=True,
            )
        else:
            self.patch_drop = None

        if use_rot_pos_emb:
            ref_feat_shape = to_2tuple(ref_feat_shape) if ref_feat_shape is not None else None
            self.rope = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=self.patch_embed.grid_size,
                ref_feat_shape=ref_feat_shape,
            )

            self.rope_heatmap = RotaryEmbeddingCat(
                embed_dim // num_heads,
                in_pixels=False,
                feat_shape=self.patch_embed_heatmap.grid_size,
                ref_feat_shape=ref_feat_shape,
            )

        else:
            self.rope = None
        
        self.depth = depth
        self.depth_heatmap = len(depth_heatmap_idxs)
        self.depth_heatmap_idxs = depth_heatmap_idxs

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        dpr_heatmap = [x.item() for x in torch.linspace(0, drop_path_rate, self.depth_heatmap)]  # stochastic depth decay rule

        block_fn = EvaBlockPostNorm if use_post_norm else EvaBlock
        self.blocks = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(depth)])
        
        self.blocks_heatmap = nn.ModuleList([
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                qkv_bias=qkv_bias,
                qkv_fused=qkv_fused,
                mlp_ratio=mlp_ratio,
                swiglu_mlp=swiglu_mlp,
                scale_mlp=scale_mlp,
                scale_attn_inner=scale_attn_inner,
                proj_drop=proj_drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=dpr_heatmap[i],
                norm_layer=norm_layer,
                init_values=init_values,
            )
            for i in range(self.depth_heatmap)])

        self.norm = norm_layer(embed_dim) if last_norm else nn.Identity()

        ## three layer mlp for classification
        self.head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, num_classes)
        )

        if self.pos_embed is not None:
            trunc_normal_(self.pos_embed, std=.02)

        if self.pos_embed_heatmap is not None:
            trunc_normal_(self.pos_embed_heatmap, std=.02)
        
        return

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)
        
        for layer_id, layer in enumerate(self.blocks_heatmap):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        ## init weights
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        self.apply(_init_weights)
        self.fix_init_weight()

        ## load pretrained weights
        super().init_weights(pretrained)

        return

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed'}

    def forward_features(self, x, heatmap=None):
        B, C, H, W = x.shape
        x, (Hp, Wp) = self.patch_embed(x) ## x is [B, num_patches, embed_dim]

        x_heatmap, _ = self.patch_embed_heatmap(heatmap) ## x is [B, num_patches, embed_dim]

        ###------------------------------------------------------------------------------------
        if self.pos_embed is not None:
             # fit for multiple GPU training
            # since the first element for pos embed (sin-cos manner) is zero, it will cause no difference
            ## this is the vitpose trick. TODO: check if it is necessary, why not x = x + self.pos_embed    
            x = x + self.pos_embed[:, 1:] + self.pos_embed[:, :1]

        if self.pos_embed_heatmap is not None:
            x_heatmap = x_heatmap + self.pos_embed_heatmap[:, 1:] + self.pos_embed_heatmap[:, :1]
        
        ## if using eva02, class token is concatenated to the front of x and used in prediction.
        ## otherwise the rotation position embedding does not work
        if self.cls_token is not None:
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        if self.cls_token_heatmap is not None:
            x_heatmap = torch.cat((self.cls_token_heatmap.expand(x_heatmap.shape[0], -1, -1), x_heatmap), dim=1)

        x = self.pos_drop(x)

        # obtain shared rotary position embedding and apply patch dropout
        rot_pos_embed = self.rope.get_embed() if self.rope is not None else None ## 768 x 128
        rot_pos_embed_heatmap = self.rope_heatmap.get_embed() if self.rope_heatmap is not None else None ## 768 x 128

        if self.patch_drop is not None:
            x, keep_indices = self.patch_drop(x)
            if rot_pos_embed is not None and keep_indices is not None:
                rot_pos_embed = apply_keep_indices_nlc(x, rot_pos_embed, keep_indices)
        
        blk_heatmap_idx = 0
        for idx, blk in enumerate(self.blocks):
            x = blk(x, rope=rot_pos_embed)

            ## inject information from heatmap
            if idx in self.depth_heatmap_idxs:
                x_heatmap = self.blocks_heatmap[blk_heatmap_idx](x_heatmap, rope=rot_pos_embed_heatmap)
                x = x + x_heatmap
                blk_heatmap_idx += 1

        x = self.norm(x)

        ## extract the class token
        cls_token = x[:, 0]

        return cls_token
    
    def forward_head(self, x):
        x = self.head(x)
        return x

    def forward(self, x, heatmap=None):
        _, _, H, W = x.shape

        if heatmap is not None:
            heatmap = F.interpolate(heatmap, size=(H, W), mode='bilinear', align_corners=False) ## B x 1 x H x W

        x = self.forward_features(x, heatmap=heatmap) ## returns the class token
        y = self.forward_head(x)

        return y

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
