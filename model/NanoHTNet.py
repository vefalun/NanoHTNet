## Our PoseFormer model was revised from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
# Written by Ce Zheng (cezheng@knights.ucf.edu)
# Modified by Qitao Zhao (qitaozhao@mail.sdu.edu.cn)

import math
import logging
from functools import partial
from einops import rearrange

import torch
import torch_dct as dct
import torch.nn as nn
import torch.nn.functional as F


import numpy as np
from timm.models.layers import DropPath
from model.Block import Hiremixer, Hiremixer_frame
from common.opt import opts
opt = opts().parse()




class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x



  
class NanoHTNet(nn.Module):
    def __init__(self, args, adj, adj_tem):
        super().__init__()

        layers, channel, d_hid, self.length  = args.layers, args.channel, args.d_hid, args.frames
        self.batch, self.num_joints_in, self.num_joints_out = args.batch_size, args.n_joints, args.out_joints
        self.num_frame_kept = opt.keep_frames
        self.f = opt.frames
        
        self.tem_patch_embed = nn.Linear(2*self.num_joints_in, channel)
        self.tem_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, channel))
        self.tem_Transformer = Hiremixer_frame(adj_tem, layers, channel, d_hid, length=self.num_frame_kept)
        self.fcn_tem = nn.Linear(args.channel, self.num_joints_out * opt.hid)
        
        self.patch_embed = nn.Linear(2*self.length, channel)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, channel))
        self.Transformer = Hiremixer(adj, layers, channel, d_hid, length=self.num_frame_kept)
        self.fcn_spa = nn.Linear(args.channel, opt.hid*self.num_frame_kept)

        if args.pretrain:
            self.fc = nn.Sequential(
                nn.LayerNorm(opt.hid*2),
                nn.Linear(opt.hid*2 , args.feature_dim),
                nn.LayerNorm(args.feature_dim),
                nn.Conv2d(in_channels=9, out_channels=1, kernel_size=(1, 1), stride=(1, 1)),
                nn.ReLU(),
            )
        else:     
            self.fc = nn.Linear(opt.hid*2, 3)
        
        

    def forward(self, x):
        x_tem = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :self.num_frame_kept]
        x_tem = rearrange(x_tem, 'b j c f -> b f (j c)').contiguous()
        x_tem = self.tem_patch_embed(x_tem)
        x_tem = x_tem + self.tem_pos_embed
        x_tem = self.tem_Transformer(x_tem)
        x_tem = self.fcn_tem(x_tem)
        x_tem = rearrange(x_tem, 'b f (j c) -> b f j c', j=self.num_joints_out).contiguous()
        
        x_spa = rearrange(x, 'b f j c -> b j (f c)', j=self.num_joints_out).contiguous()
        x_spa = self.patch_embed(x_spa)
        x_spa = x_spa + self.pos_embed
        x_spa = self.Transformer(x_spa)
        x_spa = self.fcn_spa(x_spa)
        x_spa = rearrange(x_spa, 'b j (f c) -> b f j c',f=self.num_frame_kept).contiguous()
        
        x = torch.cat((x_spa, x_tem), dim=-1)
        x = self.fc(x)
        return x