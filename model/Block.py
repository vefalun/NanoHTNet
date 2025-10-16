from functools import partial
import torch
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.layers import DropPath
from model.GCN_conv import ModulatedGraphConv
from model.Transformer import Attention, Mlp
from einops.einops import rearrange

#X_1
rl_2joints = [2,3]
ll_2joints = [5,6]
la_2joints = [12,13]
ra_2joints = [15,16]
part_2joints = [rl_2joints,ll_2joints,la_2joints,ra_2joints]
# X_2
rl_3joints = [1,2,3]
ll_3joints = [4,5,6]
ra_3joints = [14,15,16]
la_3joints = [11,12,13]
part_3joints = [rl_3joints,ll_3joints,la_3joints,ra_3joints]

class LJC(nn.Module):
    def __init__(self, adj, dim, drop_path=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.adj = adj
        self.norm_gcn1 = norm_layer(dim)
        self.gcn1 = ModulatedGraphConv(dim,384,self.adj)
        self.gelu = nn.GELU()
        self.gcn2 = ModulatedGraphConv(384,dim,self.adj)
        self.norm_gcn2 = norm_layer(dim)

    def forward(self, x_gcn):
        x_gcn = x_gcn + self.drop_path(self.norm_gcn2(self.gcn2(self.gelu(self.gcn1(self.norm_gcn1(x_gcn))))))
        return x_gcn


class IPC(nn.Module):
    def __init__(self, dim, mlp_hidden_dim, drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.index_1 =    [1,2,3, 4,5,6, 11,12,13, 14,15,16] # 6parts
        self.index_2 =    [2,3, 5,6, 12,13, 15,16]
        self.gelu = nn.GELU()
        self.norm_conv1 = norm_layer(dim)
        self.conv1 = nn.Conv1d(dim,dim, kernel_size=3, padding=0, stride=3)
        self.norm_conv1_mlp = norm_layer(dim)
        self.mlp_down_1 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.norm_conv2 = norm_layer(dim)
        self.conv2 = nn.Conv1d(dim,dim, kernel_size=2, padding=0, stride=2)
        self.norm_conv2_mlp = norm_layer(dim)
        self.mlp_down_2 = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)


    def forward(self,  x_gcn, x_conv):
        x_conv = x_conv + x_gcn
        
        #NOTE:Conv_1  3 joints per limb
        x_conv_1 = self.norm_conv1(x_conv)
        x_conv_1 = x_conv_1.permute(0,2,1)
        x_pooling_1 = x_conv_1[:, :, self.index_1]
        x_pooling_1 = self.drop_path(self.gelu(self.conv1(x_pooling_1)))

        x_pooling_1 = x_pooling_1.permute(0,2,1)
        x_pooling_1 = x_pooling_1 + self.drop_path(self.mlp_down_1(self.norm_conv1_mlp(x_pooling_1)))
        x_pooling_1 = x_pooling_1.permute(0,2,1)
        for i in range(len(part_3joints)):
            num_joints = len(part_3joints[i]) - 1
            x_conv_1[:,:,part_3joints[i][1:]] = x_pooling_1[:,:,i].unsqueeze(-1).repeat(1,1,num_joints)
        x_conv_1 = x_conv_1.permute(0,2,1)

        #NOTE:Conv_2  2 joints per limb
        x_conv_2 = self.norm_conv2(x_conv)
        x_conv_2 = x_conv_2.permute(0,2,1)
        x_pooling_2 = x_conv_2[:, :, self.index_2]
        x_pooling_2 = self.drop_path(self.gelu(self.conv2(x_pooling_2)))

        x_pooling_2 = x_pooling_2.permute(0,2,1)
        x_pooling_2 = x_pooling_2 + self.drop_path(self.mlp_down_2(self.norm_conv2_mlp(x_pooling_2)))
        x_pooling_2 = x_pooling_2.permute(0,2,1)
        for i in range(len(part_2joints)):
            num_joints = len(part_2joints[i]) - 1
            x_conv_2[:,:,part_2joints[i][1:]] = x_pooling_2[:,:,i].unsqueeze(-1).repeat(1,1,num_joints)
        x_conv_2 = x_conv_2.permute(0,2,1)

        x_conv = x_conv_1 + x_conv_2 + x_conv
        return x_conv


class GBI(nn.Module):
    def __init__(self, dim, num_heads,  qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, length=1):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_attn = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, length=length)

    def forward(self, x_conv, x_attn):
        x_attn = x_attn + x_conv
        x_attn = x_attn + self.drop_path(self.attn(self.norm_attn(x_attn)))
        return x_attn





class Hiremixer(nn.Module):
    def __init__(self, adj, depth=8, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.15, length=9):
        super().__init__()
        drop_path_rate = 0.3
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                adj, dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, length=length)
            for i in range(depth)])
        self.Temporal_norm = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        return x


class Block(nn.Module):
    def __init__(self, adj, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,    norm_layer=nn.LayerNorm, length=1):
        super().__init__()

        dim = int(dim/3)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Three sub-modules
        self.lgc = LJC(adj, dim, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.ipc = IPC(dim, mlp_hidden_dim, drop=drop, drop_path=drop_path, act_layer=nn.GELU, norm_layer=nn.LayerNorm)
        self.gbi = GBI(dim, num_heads,  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,
                 drop_path=drop_path, norm_layer=nn.LayerNorm, length=length)

        self.norm_mlp = norm_layer(dim*3)
        self.mlp = Mlp(in_features=dim*3, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)



    def forward(self, x):
        x_split = torch.chunk(x,3,-1)
        x_lgc, x_ipc, x_gbi = x_split
        # Local Joint-level Connection (LJC)
        x_lgc = self.lgc(x_lgc)
        # Inter-Part Constraint (IPC)
        x_ipc = self.ipc(x_lgc, x_ipc)
        # Global body-level Interaction (GBI)
        x_gbi = self.gbi(x_ipc, x_gbi)
        x_cat = torch.cat([x_lgc,x_ipc,x_gbi], -1)
        x = x_cat + self.drop_path(self.mlp(self.norm_mlp(x_cat)))
        return x



class Hiremixer_frame(nn.Module):
    def __init__(self, adj, depth=8, embed_dim=512, mlp_hidden_dim=1024, h=8, drop_rate=0.1, length=9):
        super().__init__()
        drop_path_rate = 0.3
        attn_drop_rate = 0.
        qkv_bias = True
        qk_scale = None
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0.1, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block_frame(
                adj, dim=embed_dim, num_heads=h, mlp_hidden_dim=mlp_hidden_dim, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, length=length)
            for i in range(depth)])
        self.Temporal_norm = norm_layer(embed_dim)

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        x = self.Temporal_norm(x)
        return x


class Block_frame(nn.Module):
    def __init__(self, adj, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU,    norm_layer=nn.LayerNorm, length=1):
        super().__init__()

        dim = int(dim/2)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        # Three sub-modules
        self.lgc = LJC(adj, dim, drop_path=drop_path, norm_layer=nn.LayerNorm)
        self.gbi = GBI(dim, num_heads,  qkv_bias=qkv_bias, qk_scale=qk_scale, drop=0.1, attn_drop=attn_drop,
                 drop_path=drop_path, norm_layer=nn.LayerNorm, length=length)

        self.norm_mlp = norm_layer(dim*2)
        self.mlp = Mlp(in_features=dim*2, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)



    def forward(self, x):
        x_split = torch.chunk(x,2,-1)
        x_lgc,  x_gbi = x_split
        # Local Joint-level Connection (LJC)
        x_lgc = self.lgc(x_lgc)
        # Global body-level Interaction (GBI)
        x_gbi = self.gbi(x_lgc, x_gbi)
        x_cat = torch.cat([x_lgc,x_gbi], -1)
        x = x_cat + self.drop_path(self.mlp(self.norm_mlp(x_cat)))
        return x
    







class Block_tem(nn.Module):
    def __init__(self, dim, num_heads, mlp_hidden_dim, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, depth=0):
        super().__init__()

        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, \
            qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x
    





