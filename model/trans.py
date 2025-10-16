import sys
from einops.einops import rearrange


sys.path.append("..")
import torch
import torch.nn as nn
from model.Block import Hiremixer, Hiremixer_frame, Block, Block_frame
from model.Block import Block_tem
from functools import partial
from timm.models.layers import DropPath
import torch_dct as dct
from common.opt import opts
opt = opts().parse()



class HTNet(nn.Module):
    def __init__(self, args, adj):
        super().__init__()

        if args == -1:
            layers, channel, d_hid, length  = 3, 512, 1024, 27
            self.num_joints_in, self.num_joints_out = 17, 17
        else:
            layers, channel, d_hid, length  = args.layers, args.channel, args.d_hid, args.frames
            self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.patch_embed = nn.Linear(2, channel)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, channel))
        self.Hiremixer = Hiremixer(adj, layers, channel, d_hid, length=length)

        if args.pretrain:
            self.fc = nn.Sequential(
                nn.LayerNorm(channel),
                nn.Linear(channel , args.feature_dim),
                nn.ReLU(),
            )
        elif args.freeze:
            self.fc = nn.Sequential(
                nn.LayerNorm(channel),
                nn.Linear(channel , 512),
                nn.ReLU(),
                nn.LayerNorm(512),
                nn.Linear(512 , channel),
                nn.ReLU(),
                nn.Linear(args.channel, 3)
            )             
        else:     
            self.fc = nn.Linear(args.channel, 3)



    def forward(self, x):
        x = rearrange(x, 'b f j c -> (b f) j c').contiguous()
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.Hiremixer(x)
        x = self.fc(x)
        x = x.view(x.shape[0], -1, self.num_joints_out, x.shape[2])
        return x






class HTNet_f(nn.Module):
    def __init__(self, args, adj, adj_tem):
        super().__init__()

        if args == -1:
            layers, channel, d_hid, self.length  = 3, 512, 1024, 27
            self.num_joints_in, self.num_joints_out = 17, 17
        else:
            layers, channel, d_hid, self.length  = args.layers, args.channel, args.d_hid, args.frames
            self.num_joints_in, self.num_joints_out = args.n_joints, args.out_joints

        self.patch_embed = nn.Linear(2*self.length, channel)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_joints_in, channel))
        self.Transformer = Hiremixer(adj, layers, channel, d_hid, length=self.length)
        self.fcn_spa = nn.Linear(args.channel, 2*self.length)
        
        self.tem_patch_embed = nn.Linear(2*self.num_joints_in, channel)
        self.tem_pos_embed = nn.Parameter(torch.zeros(1, self.length, channel))
        self.tem_Transformer = Hiremixer_frame(adj_tem, layers, channel, d_hid, length=self.length)
        
        self.fcn = nn.Linear(args.channel, 3*self.num_joints_out)
        
        

    def forward(self, x):
        x = rearrange(x, 'b f j c -> b j (f c)').contiguous()
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.Transformer(x)
        x = self.fcn_spa(x)
        
        
        x = rearrange(x, 'b j (f c) -> b f (j c)',f = self.length).contiguous()
        x = self.tem_patch_embed(x)
        x = x + self.tem_pos_embed
        x = self.tem_Transformer(x)
        
        x = self.fcn(x)
        x = rearrange(x, 'b f (j c) -> b f j c',j=self.num_joints_out).contiguous()
        # x = x.view(x.shape[0], -1, self.num_joints_out, 3)
        return x
    


class Mixste_htnet(nn.Module):
    def __init__(self, opt, adj, adj_tem):
        super().__init__()

        depth = opt.layers
        channel = opt.channel
        d_hid = opt.d_hid
        frames  =  opt.frames
        
        drop_path_rate = 0.1 # 0.2
        drop_rate = 0.

        num_joints = 17

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.Spatial_patch_to_embedding = nn.Linear(2, channel)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, channel))
        #TODO: 注意修改
        # self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, opt.frames, channel))
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, opt.stride, channel))

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.block_depth = depth
        
        self.STEblocks = nn.ModuleList([ Hiremixer(adj, 1, channel, d_hid, length=frames)
            for i in range(depth)])

        if opt.attempt == 0:
            self.TTEblocks = nn.ModuleList([ Hiremixer_frame(adj_tem, 1, channel, d_hid, length=frames)
                for i in range(depth)])
        elif opt.attempt == 1:
            self.TTEblocks = nn.ModuleList([
                Block_tem(
                    dim=channel, num_heads=8, mlp_hidden_dim=d_hid, qkv_bias=True, qk_scale=None,
                    drop=0.1, attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, depth=depth)
                for i in range(depth)])


        self.Spatial_norm = norm_layer(channel)
        self.Temporal_norm = norm_layer(channel)

        if opt.pretrain:
            self.fc = nn.Sequential(
                nn.LayerNorm(channel),
                nn.Linear(channel , opt.feature_dim),
            )       
        else:     
            self.fc = nn.Sequential(
                nn.LayerNorm(channel),
                nn.Linear(channel , 3),
            )

    def forward(self, x):
        b, f, n, c = x.shape

        ## STE_forward
        x = rearrange(x, 'b f n c  -> (b f) n c')
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)

        ## TTE_foward
        x = rearrange(x, '(b f) n c -> (b n) f c', f=f)
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        x = self.TTEblocks[0](x)
        if opt.attempt == 1:
            x = self.Temporal_norm(x)

        ## ST_foward
        x = rearrange(x, '(b n) f c -> b f n c', n=n)
        for i in range(1, self.block_depth):
            x = rearrange(x, 'b f n c -> (b f) n c')
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = steblock(x)
            x = rearrange(x, '(b f) n c -> (b n) f c', f=f)

            x = tteblock(x)
            if opt.attempt == 1:
                x = self.Temporal_norm(x)
            x = rearrange(x, '(b n) f c -> b f n c', n=n)

        x = self.fc(x)

        x = x.view(b, f, n, -1)

        return x
    



class Mixhtnet(nn.Module):
    def __init__(self, opt, adj, adj_tem):
        super().__init__()

        depth = opt.layers
        channel = opt.channel
        d_hid = opt.d_hid
        frames  =  opt.frames
        drop_path_rate = 0.1 # 0.2
        drop_rate = 0.
        num_joints = 17
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        
        
        self.block_depth = depth
        self.num_frame_kept = opt.keep_frames
        self.Spatial_patch_to_embedding = nn.Linear(2*self.num_frame_kept, channel)
        self.Spatial_pos_embed = nn.Parameter(torch.zeros(1, num_joints, channel))
        #TODO: 注意修改
        # self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, opt.frames, channel))
        print(channel,  self.num_frame_kept)
        self.Temporal_pos_embed = nn.Parameter(torch.zeros(1, self.num_frame_kept, (channel // self.num_frame_kept)*17))
        self.pos_drop = nn.Dropout(p=drop_rate)

        
        self.STEblocks = nn.ModuleList([ 
                Block(
                    adj, dim=channel, num_heads=8, mlp_hidden_dim=d_hid, qkv_bias=True, qk_scale=None,
                    drop=drop_rate, attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, length=self.num_frame_kept)
            for i in range(depth)])

        self.TTEblocks = nn.ModuleList([
            Block_frame(
                adj_tem, dim=channel, num_heads=8, mlp_hidden_dim=d_hid, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0., drop_path=dpr[i], norm_layer=norm_layer, length=self.num_frame_kept)
            for i in range(depth)])



        self.Spatial_norm = norm_layer(channel)
        self.Temporal_norm = norm_layer(channel)
        self.Spatial_fcn = nn.Linear(channel, opt.frames)

        if opt.pretrain:
            self.fc = nn.Sequential(
                nn.LayerNorm(channel),
                nn.Linear(channel , opt.feature_dim),
            )       
        else:     
            self.fc = nn.Sequential(
                nn.LayerNorm(channel),
                nn.Linear(channel , 3),
            )

    def forward(self, x):
        x = dct.dct(x.permute(0, 2, 3, 1))[:, :, :, :self.num_frame_kept]
        b, j, c, f = x.shape

        ## STE_forward
        x = rearrange(x, 'b j c f  -> b j (f c)')#/27, 144   432
        x = self.Spatial_patch_to_embedding(x)
        x += self.Spatial_pos_embed
        x = self.pos_drop(x)
        x = self.STEblocks[0](x)
        x = self.Spatial_norm(x)

        ## TTE_foward
        x = rearrange(x, 'b j (f c) -> b f (j c)', f=f) #272
        x += self.Temporal_pos_embed
        x = self.pos_drop(x)
        print(x.shape)
        x = self.TTEblocks[0](x)
        x = self.Temporal_norm(x)

        ## ST_foward
        x = rearrange(x, 'b f (j c) -> b f j c', n=j)
        for i in range(1, self.block_depth):
            steblock = self.STEblocks[i]
            tteblock = self.TTEblocks[i]
            
            x = rearrange(x, 'b f j c -> b j (f c)')
            x = steblock(x)
            x = self.Spatial_norm(x)
            
            x = rearrange(x, 'b j (f c) -> b f (j c)', f=f)
            x = tteblock(x)
            x = self.Temporal_norm(x)
            x = rearrange(x, 'b f (j c) -> b f j c', n=j)
            print(x.shape)

        x = self.fc(x)

        x = x.view(b, f, j, -1)

        return x
    

