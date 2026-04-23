# Code Implementation of the MambaIR Model
import math
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
import torch.nn.functional as F
from functools import partial
from typing import Optional, Callable


from kornia.core.external import numpy

from basicsr.utils.registry import ARCH_REGISTRY
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mambamain.mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from einops import rearrange, repeat




NEG_INF = -1000000


class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class CAB(nn.Module):
    def __init__(self, num_feat, is_light_sr= False, compress_ratio=3,squeeze_factor=30):
        super(CAB, self).__init__()
        if is_light_sr: # a larger compression ratio is used for light-SR
            compress_ratio = 6
        self.cab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat // compress_ratio, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(num_feat // compress_ratio, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor)
        )

    def forward(self, x):
        return self.cab(x)


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


class DynamicPosBias(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, biases):
        pos = self.pos3(self.pos2(self.pos1(self.pos_proj(biases))))
        return pos

    def flops(self, N):
        flops = N * 2 * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.pos_dim
        flops += N * self.pos_dim * self.num_heads
        return flops


class Attention(nn.Module):
    r""" Multi-head self attention module with dynamic position bias.

    Args:
        dim (int): Number of input channels.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.,
                 position_bias=True):

        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.position_bias = position_bias
        if self.position_bias:
            self.pos = DynamicPosBias(self.dim // 4, self.num_heads)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, H, W, mask=None):
        """
        Args:
            x: input features with shape of (num_groups*B, N, C)
            mask: (0/-inf) mask with shape of (num_groups, Gh*Gw, Gh*Gw) or None
            H: height of each group
            W: width of each group
        """
        group_size = (H, W)
        B_, N, C = x.shape
        assert H * W == N
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # (B_, self.num_heads, N, N), N = H*W

        if self.position_bias:
            # generate mother-set
            position_bias_h = torch.arange(1 - group_size[0], group_size[0], device=attn.device)
            position_bias_w = torch.arange(1 - group_size[1], group_size[1], device=attn.device)
            biases = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))  # 2, 2Gh-1, 2W2-1
            biases = biases.flatten(1).transpose(0, 1).contiguous().float()  # (2h-1)*(2w-1) 2

            # get pair-wise relative position index for each token inside the window
            coords_h = torch.arange(group_size[0], device=attn.device)
            coords_w = torch.arange(group_size[1], device=attn.device)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Gh, Gw
            coords_flatten = torch.flatten(coords, 1)  # 2, Gh*Gw
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Gh*Gw, Gh*Gw
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Gh*Gw, Gh*Gw, 2
            relative_coords[:, :, 0] += group_size[0] - 1  # shift to start from 0
            relative_coords[:, :, 1] += group_size[1] - 1
            relative_coords[:, :, 0] *= 2 * group_size[1] - 1
            relative_position_index = relative_coords.sum(-1)  # Gh*Gw, Gh*Gw

            pos = self.pos(biases)  # 2Gh-1 * 2Gw-1, heads
            # select position bias
            relative_position_bias = pos[relative_position_index.view(-1)].view(
                group_size[0] * group_size[1], group_size[0] * group_size[1], -1)  # Gh*Gw,Gh*Gw,nH
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Gh*Gw, Gh*Gw
            attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nP = mask.shape[0]
            attn = attn.view(B_ // nP, nP, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(
                0)  # (B, nP, nHead, N, N)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

##local_scan代码
def local_scan(x, w=8, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input: 
        x: [B, L, C]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, L, C = x.shape
    x = x.view(B, H, W, C)
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if H % w != 0 or W % w != 0:
        newH, newW = Hg * w, Wg * w
        x = F.pad(x, (0, 0, 0, newW - W, 0, newH - H))
    if column_first:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 5, 3, 1, 4, 2).reshape(B, C, -1)
    else:
        x = x.view(B, Hg, w, Wg, w, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, -1)
    if flip:
        x = x.flip([-1])
    return x

def local_reverse(x, w=8, H=14, W=14, flip=False, column_first=False):
    """Local windowed scan in LocalMamba
    Input: 
        x: [B, C, L]
        H, W: original width and height before padding
        column_first: column-wise scan first (the additional direction in VMamba)
    Return: [B, C, L]
    """
    B, C, L = x.shape
    Hg, Wg = math.ceil(H / w), math.ceil(W / w)
    if flip:
        x = x.flip([-1])
    if H % w != 0 or W % w != 0:
        if column_first:
            x = x.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, Hg * w, Wg * w)
        else:
            x = x.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, Hg * w, Wg * w)
        x = x[:, :, :H, :W].reshape(B, C, -1)
    else:
        if column_first:
            x = x.view(B, C, Wg, Hg, w, w).permute(0, 1, 3, 5, 2, 4).reshape(B, C, L)
        else:
            x = x.view(B, C, Hg, Wg, w, w).permute(0, 1, 2, 4, 3, 5).reshape(B, C, L)
    return x
#下采样，由level决定# 信息从空间维度转移到批次维度
# [B, C, H, W] → [B, C, H//r, r, W//r, r] → [B*r², C, H//r, W//r]

def get_sample_img(x,h,w,level=1):
    ratio=2**level
    if(h%ratio!=0 or w%ratio!=0):
        newh,neww=math.ceil(h/ratio)*ratio,math.ceil(w/ratio)*ratio
        x=F.pad(x,(0,neww-w,0,newh-h))
    B,C,H,W=x.shape
    x=x.view(B,C,H//ratio,ratio,W//ratio,ratio).permute(0,3,5,1,2,4).contiguous()
    x=x.view(-1,C,H//ratio,W//ratio)
    return x
# 上采样，信息从批次维度转移回空间维度
# [B*r², C, H//r, W//r] → [B, r, r, C, H//r, W//r] → [B, C, H, W]
def reverse_sample_img(y,h,w,level=1):
    KB,C,H,W=y.shape
    ratio=2**level
    y=y.view(-1,ratio,ratio,C,H,W).permute(0,3,4,1,5,2).contiguous()
    y=y.view(-1,C,H*ratio,W*ratio)
    if(h%ratio!=0 or w%ratio!=0):
        y=y[:,:,:h,:w]
    return y




class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            multi_scale=True,
            parallel=True,
            level_reverse=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.multi_scale = multi_scale
        self.parallel = parallel
        self.level_reverse = level_reverse
    

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        num_groups = 2 if multi_scale else 1
        param_groups = [
            self.init_group_params(
                d_inner=self.d_inner,
                d_state=self.d_state, 
                dt_rank=self.dt_rank,
                dt_scale=dt_scale,
                dt_init=dt_init,
                dt_min=dt_min,
                dt_max=dt_max,
                dt_init_floor=dt_init_floor,
                device=device,
                dtype=dtype
            ) for _ in range(num_groups)
        ]

        for i, (x_proj_w, dt_projs_w, dt_projs_b, a_logs, ds) in enumerate(param_groups):
            setattr(self, f'x_proj_weight_{i}', x_proj_w)
            setattr(self, f'dt_projs_weight_{i}', dt_projs_w) 
            setattr(self, f'dt_projs_bias_{i}', dt_projs_b)
            setattr(self, f'A_logs_{i}', a_logs)
            setattr(self, f'Ds_{i}', ds)

        self.selective_scan = selective_scan_fn
        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

        # self.fusion_conv = nn.Conv2d(self.d_inner * 3, self.d_inner, kernel_size=1)

    def init_group_params(
        self, 
        d_inner=None,      
        d_state=None,      
        dt_rank=None,      
        copies=4,          
        dt_scale=1.0,      
        dt_init="random",  
        dt_min=0.001,      
        dt_max=0.1,        
        dt_init_floor=1e-4,
        device=None,       
        dtype=None,        
        merge=True         
    ):
        """
        
        Returns:
            tuple: (x_proj_weight, dt_projs_weight, dt_projs_bias, A_logs, Ds)
        """
        d_inner = d_inner if d_inner is not None else self.d_inner
        d_state = d_state if d_state is not None else self.d_state
        dt_rank = dt_rank if dt_rank is not None else self.dt_rank

        x_proj_weight = self.x_proj_init(
            d_inner, dt_rank, d_state,
            copies=copies, device=device, dtype=dtype
        )
        
        dt_projs_weight, dt_projs_bias = self.dt_projs_init(
            dt_rank, d_inner,
            dt_scale=dt_scale, dt_init=dt_init,
            dt_min=dt_min, dt_max=dt_max,
            dt_init_floor=dt_init_floor,
            copies=copies, device=device, dtype=dtype
        )
        
        A_logs = self.A_log_init(
            d_state, d_inner, 
            copies=copies, device=device, merge=merge
        )
        
        Ds = self.D_init(
            d_inner, 
            copies=copies, device=device, merge=merge
        )
        
        return x_proj_weight, dt_projs_weight, dt_projs_bias, A_logs, Ds

    @staticmethod
    def x_proj_init(d_inner, dt_rank, d_state, copies=1, device=None, dtype=None, merge=True):
        """Initialize x projection parameters
        Args:
            d_inner: inner dimension
            dt_rank: delta rank
            d_state: state dimension
            copies: number of copies (default 1)
            device: torch device
            dtype: torch dtype
            merge: whether to merge copies into one parameter
        Returns:
            nn.Parameter: x projection weights
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        x_projs = [
            nn.Linear(d_inner, (dt_rank + d_state * 2), bias=False, **factory_kwargs)
            for _ in range(copies)
        ]
        x_proj_weight = torch.stack([t.weight for t in x_projs], dim=0)  # (copies, N, inner)
        return nn.Parameter(x_proj_weight)

    @staticmethod 
    def dt_projs_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, 
                      dt_max=0.1, dt_init_floor=1e-4, copies=1, device=None, dtype=None, merge=True):
        """Initialize delta projection parameters
        Args:
            dt_rank: delta rank
            d_inner: inner dimension
            dt_scale: scale factor for initialization
            dt_init: initialization type ("constant" or "random")
            dt_min: minimum delta value
            dt_max: maximum delta value 
            dt_init_floor: minimum floor value
            copies: number of copies
            device: torch device
            dtype: torch dtype
            merge: whether to merge copies
        Returns:
            tuple(nn.Parameter, nn.Parameter): weights and biases
        """
        factory_kwargs = {"device": device, "dtype": dtype}
        dt_projs = [
            SS2D.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(copies)
        ]
        dt_projs_weight = torch.stack([t.weight for t in dt_projs], dim=0)  # (copies, inner, rank)
        dt_projs_bias = torch.stack([t.bias for t in dt_projs], dim=0)  # (copies, inner)
        
        return nn.Parameter(dt_projs_weight), nn.Parameter(dt_projs_bias)

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log
    
    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @staticmethod
    def forward_core(x: torch.Tensor, x_proj_weight, dt_projs_weight, dt_projs_bias, A_logs, Ds, d_state, dt_rank):
        """Static forward core computation
        Args:
            x: input tensor [B, C, H, W]
            x_proj_weight: x projection weights
            dt_projs_weight: delta projection weights
            dt_projs_bias: delta projection biases
            A_logs: A matrix logs
            Ds: D parameters
            d_state: state dimension
            dt_rank: delta rank
        Returns:
            tuple: (y1, y2, y3, y4) output tensors
        """
        B, C, H, W = x.shape
        L = H * W
        K = 4
        
        # h v stacking
        x_hwwh = torch.stack([
            x.contiguous().view(B, -1, L), 
            torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)
        ], dim=1).view(B, 2, -1, L)
        
        # h v hf vf concatenation
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (B, 4, C, L)
        xs = xs.permute(0, 1, 3, 2) # (B, 4, L, C)

        # Local scan
        xs = torch.cat([
            local_scan(xs[:, i], H=H if i % 2 == 0 else W, W=W if i % 2 == 0 else H).unsqueeze(1)
            for i in range(K)
        ], dim=1)  # (B, K, C, new_L)

        new_L = xs.shape[-1]

        # Projections and splits
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, new_L), x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [dt_rank, d_state, d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, new_L), dt_projs_weight)

        # Prepare inputs for selective scan
        xs = xs.float().view(B, -1, new_L)
        dts = dts.contiguous().float().view(B, -1, new_L) # (b, k * d, new_L)
        Bs = Bs.float().view(B, K, -1, new_L)
        Cs = Cs.float().view(B, K, -1, new_L) # (b, k, d_state, new_L)
        Ds = Ds.float().view(-1)
        As = -torch.exp(A_logs.float()).view(-1, d_state)
        dt_projs_bias = dt_projs_bias.float().view(-1) # (k * d)

        # Selective scan
        out_y = selective_scan_fn(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, new_L)

        # Local reverse
        out_y = torch.cat([
            local_reverse(out_y[:, i], H=H if i % 2 == 0 else W, W=W if i % 2 == 0 else H).unsqueeze(1)
            for i in range(K)
        ], dim=1)  # (B, K, C, L)

        # Final transformations
        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        
        def process_group(x, scan_level):
            x_proj_weight = getattr(self, f'x_proj_weight_{scan_level}')
            dt_projs_weight = getattr(self, f'dt_projs_weight_{scan_level}')
            dt_projs_bias = getattr(self, f'dt_projs_bias_{scan_level}')
            A_logs = getattr(self, f'A_logs_{scan_level}')
            Ds = getattr(self, f'Ds_{scan_level}')
            
            y1, y2, y3, y4 = self.forward_core(
                x, x_proj_weight, dt_projs_weight, dt_projs_bias, A_logs, Ds, self.d_state, self.dt_rank
            )
            
            assert y1.dtype == torch.float32
            return y1 + y2 + y3 + y4

        B, C, H, W = x.shape
        
        if not self.multi_scale:
            y = process_group(x, 0)
            y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
        else:
            if not self.parallel:
                if not self.level_reverse:
                    y = process_group(x, 0)
                    
                    x_level1 = get_sample_img(y.view(B, C, H, W), H, W, level=1)
                    B1, C1, H1, W1 = x_level1.shape
                    y = process_group(x_level1, 1)
                    y = y.view(B1, C1, H1, W1)
                    y = reverse_sample_img(y, H, W, level=1)
                    
                    x_level2 = get_sample_img(y.view(B, C, H, W), H, W, level=2)
                    B2, C2, H2, W2 = x_level2.shape
                    y = process_group(x_level2, 2)
                    y = y.view(B2, C2, H2, W2)
                    y = reverse_sample_img(y, H, W, level=2)
                else:
                    x_level2 = get_sample_img(x, H, W, level=2)
                    B2, C2, H2, W2 = x_level2.shape
                    y = process_group(x_level2, 2)
                    y = y.view(B2, C2, H2, W2)
                    y = reverse_sample_img(y, H, W, level=2)
                    
                    x_level1 = get_sample_img(y.view(B, C, H, W), H, W, level=1)
                    B1, C1, H1, W1 = x_level1.shape
                    y = process_group(x_level1, 1)
                    y = y.view(B1, C1, H1, W1)
                    y = reverse_sample_img(y, H, W, level=1)
                    
                    y = process_group(y.view(B, C, H, W), 0)
                
                y = y.transpose(1, 2).contiguous().view(B, H, W, -1)
            else:
                ##level0
                y0 = process_group(x, 0)
                y0 = y0.transpose(1, 2).contiguous().view(B, H, W, -1)
                
                # level 1
                x_level1 = get_sample_img(x, H, W, level=1)
                B1, C1, H1, W1 = x_level1.shape
                y1 = process_group(x_level1, 1)
                y1 = y1.view(B1, C1, H1, W1)
                y1 = reverse_sample_img(y1, H, W, level=1)
                y1 = y1.transpose(1, 2).contiguous().view(B, H, W, -1)
                
                # level 2
                # x_level2 = get_sample_img(x, H, W, level=2)
                # B2, C2, H2, W2 = x_level2.shape
                # y2 = process_group(x_level2, 2)
                # y2 = y2.view(B2, C2, H2, W2)
                # y2 = reverse_sample_img(y2, H, W, level=2)
                # y2 = y2.transpose(1, 2).contiguous().view(B, H, W, -1)
                
                # y = (y0 + y1 + y2) / 3
                y=(y0+y1)/2

        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class VSSBlock(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 16,
            expand: float = 2.,
            is_light_sr: bool = False,
            multi_scale: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)
        self.self_attention = SS2D(
            d_model=hidden_dim, 
            d_state=d_state,
            expand=expand,
            dropout=attn_drop_rate, 
            multi_scale=multi_scale,
            **kwargs
        )
        self.drop_path = DropPath(drop_path)
        self.skip_scale = nn.Parameter(torch.ones(hidden_dim))
        self.conv_blk = CAB(hidden_dim, is_light_sr)
        self.ln_2 = nn.LayerNorm(hidden_dim)
        self.skip_scale2 = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, input, x_size):
        # x [B,HW,C]
        B, L, C = input.shape
        input = input.view(B, *x_size, C).contiguous()  # [B,H,W,C]
        x = self.ln_1(input)
        x = input * self.skip_scale + self.drop_path(self.self_attention(x))
        x = x * self.skip_scale2 + self.conv_blk(self.ln_2(x).permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        x = x.view(B, -1, C).contiguous()
        return x


class BasicLayer(nn.Module):
    """ The Basic MambaIR Layer in one Residual State Space Group
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 depth,
                 drop_path=0.,
                 d_state=16,
                 mlp_ratio=2.,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_light_sr=False,
                 multi_scale=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.mlp_ratio = mlp_ratio
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList()
        if depth > 1:
            for i in range(depth-1):
                self.blocks.append(VSSBlock(
                    hidden_dim=dim,
                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                    norm_layer=nn.LayerNorm,
                    attn_drop_rate=0,
                    d_state=d_state,
                    expand=self.mlp_ratio,
                    input_resolution=input_resolution,
                    is_light_sr=is_light_sr,
                    multi_scale=False
                ))
        self.blocks.append(VSSBlock(
                hidden_dim=dim,
                drop_path=drop_path[-1] if isinstance(drop_path, list) else drop_path,
                norm_layer=nn.LayerNorm,
                attn_drop_rate=0,
                d_state=d_state,
                expand=self.mlp_ratio,
                input_resolution=input_resolution,
                is_light_sr=is_light_sr,
                multi_scale=multi_scale
            ))
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x, x_size)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops



def index_reverse_v2(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1,
                      padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x

class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x





class ASSM(nn.Module):
    def __init__(self, dim, d_state, input_resolution, num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse_v2(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold


        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))

        x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold

        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        #  our ASE here ---
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l) #prompt
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y


class DiagnosticLogger:
    """统一的诊断日志管理器"""

    @staticmethod
    def log_tensor_stats(name, tensor, prefix=""):
        """记录tensor的统计信息"""
        if tensor is None:
            print(f"{prefix}[{name}] None")
            return

        mean_val = tensor.mean().item()
        std_val = tensor.std().item()
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        abs_mean = tensor.abs().mean().item()

        print(f"{prefix}[{name}]")
        print(f"{prefix}  Shape: {tuple(tensor.shape)}")
        print(f"{prefix}  Range: [{min_val:.6f}, {max_val:.6f}]")
        print(f"{prefix}  Mean: {mean_val:.6f}, Std: {std_val:.6f}")
        print(f"{prefix}  Abs Mean: {abs_mean:.6f}")

    @staticmethod
    def log_weights_distribution(soft_weights, prefix=""):
        """记录软权重的分布"""
        B, L, _ = soft_weights.shape
        w_core = soft_weights[:, :, 0]
        w_flare = soft_weights[:, :, 1]
        w_bg = soft_weights[:, :, 2]

        print(f"{prefix}[Soft Weights Distribution]")
        print(f"{prefix}  Core:  mean={w_core.mean():.4f}, "
              f"max={w_core.max():.4f}, "
              f"pixels>{0.5}={((w_core > 0.5).sum().item() / (B * L) * 100):.2f}%")
        print(f"{prefix}  Flare: mean={w_flare.mean():.4f}, "
              f"max={w_flare.max():.4f}, "
              f"pixels>{0.5}={((w_flare > 0.5).sum().item() / (B * L) * 100):.2f}%")
        print(f"{prefix}  BG:    mean={w_bg.mean():.4f}, "
              f"max={w_bg.max():.4f}, "
              f"pixels>{0.5}={((w_bg > 0.5).sum().item() / (B * L) * 100):.2f}%")


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x





def index_reverse_v2(idx):
    """
    生成反向索引，用于将排序后的序列还原回原始空间位置
    """
    B, L = idx.shape
    rev_idx = torch.empty_like(idx)
    rev_idx.scatter_(1, idx, torch.arange(L, device=idx.device).unsqueeze(0).expand(B, -1))
    return rev_idx


# ========================================
# 1. ⭐ 空间注意力生成器（替代原来的LPGGuidedPromptGenerator）
# ========================================

class SpatialGuidanceGenerator(nn.Module):
    """
    将 lpg_first_stage 转换为空间注意力权重

    设计思想：
    - lpg_first_stage 大 → 空间权重大 → SSM更关注这些像素
    - lpg_first_stage 小 → 空间权重小 → SSM轻度处理
    - 使用乘法调制（而非加法），相当于门控机制
    """

    def __init__(self, d_state):
        super().__init__()
        self.d_state = d_state

        # 简单的非线性映射：lpg → 空间权重
        self.spatial_encoder = nn.Sequential(
            nn.Linear(1, d_state),
            nn.Sigmoid()  # [0, 1]，0=不处理，1=完全处理
        )

    def forward(self, lpg_first, H, W):
        """
        Args:
            lpg_first: (B, 1, H, W) - 第1次LPG

        Returns:
            spatial_weight: (B, L, d_state) - 空间注意力权重 [0, 1]
        """
        B = lpg_first.shape[0]
        L = H * W

        # 展平为序列
        lpg_flat = lpg_first.reshape(B, 1, L).transpose(1, 2)  # (B, 1, L)

        # 生成空间权重
        spatial_weight = self.spatial_encoder(lpg_flat)  # (B, L, d_state)

        return spatial_weight
# ========================================
# 2. 光源保护掩码生成器（保持不变，但简化）
# ========================================



# ========================================
# 3. ⭐ 简化的自适应路由器
# ========================================
class AdaptiveRouter(nn.Module):
    """
    简化版路由器：图像特征 + lpg_first_stage
    """

    def __init__(self, dim, num_tokens=64):
        super().__init__()

        # 图像特征编码
        self.image_proj = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU()
        )

        # ⭐ LPG特征：只用 lpg_first_stage（1维）
        self.lpg_proj = nn.Sequential(
            nn.Linear(1, dim // 4),
            nn.GELU()
        )

        # 融合并路由
        self.fusion_route = nn.Sequential(
            nn.Linear(dim // 2 + dim // 4, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, lpg_first, H, W):
        """
        Args:
            x: (B, L, C) - 图像特征
            lpg_first: (B, 1, H, W)

        Returns:
            route: (B, L, num_tokens) - 路由概率
        """
        B, L, C = x.shape

        # 图像特征
        img_feat = self.image_proj(x)  # (B, L, C//2)

        # ⭐ LPG特征：直接用值
        lpg_flat = lpg_first.reshape(B, 1, L).transpose(1, 2)  # (B, L, 1)
        lpg_feat = self.lpg_proj(lpg_flat)  # (B, L, C//4)

        # 融合
        fused = torch.cat([img_feat, lpg_feat], dim=-1)
        route = self.fusion_route(fused)

        return route


# ========================================
# 4. ⭐ 带保护和空间注意力的 Selective Scan
# ========================================
class SmartProtectedSelectiveScan(nn.Module):
    """
    智能版 SSM：根据保护掩码选择性地处理

    保护策略：
    - 光源区域：直接返回输入，跳过SSM计算
    - 其他区域：正常SSM处理
    """

    def __init__(self, d_model, d_state=16, expand=2, dt_rank="auto",
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0,
                 dt_init_floor=1e-4, device=None, dtype=None):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        # 标准SSM参数
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
        self.x_proj_weight = nn.Parameter(self.x_proj.weight.unsqueeze(0))
        del self.x_proj

        self.dt_proj = self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                                    **factory_kwargs)
        self.dt_projs_weight = nn.Parameter(self.dt_proj.weight.unsqueeze(0))
        self.dt_projs_bias = nn.Parameter(self.dt_proj.bias.unsqueeze(0))
        del self.dt_proj

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)

        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward(self, x, spatial_weight=None, protection_mask=None, prompt=None):
        B, L, C = x.shape

        if protection_mask is None:
            return self._original_ssm_forward(x, spatial_weight, None, prompt)

        if protection_mask.all():
            return x

        y_full = x.clone()

        for b in range(B):
            mask_b = protection_mask[b]  # (L,)

            if mask_b.all():
                continue

            if not mask_b.any():
                y_full[b] = self._original_ssm_forward(
                    x[b:b + 1],
                    spatial_weight[b:b + 1] if spatial_weight is not None else None,
                    None,
                    prompt[b:b + 1] if prompt is not None else None
                )[0]
                continue

            unprotected_mask = ~mask_b
            unprotected_indices = unprotected_mask.nonzero(as_tuple=True)[0]  # (N,)

            x_unprot = x[b, unprotected_indices, :].unsqueeze(0)  # (1, N, C)

            # ── B_weight 和 C_weight 分别提取非保护区域 ──────────────────────
            if spatial_weight is not None:
                sw_unprot = spatial_weight[b, unprotected_indices, :].unsqueeze(0)
            else:
                sw_unprot = None
            pr_unprot = prompt[b, unprotected_indices, :].unsqueeze(0) if prompt is not None else None

            y_unprot = self._original_ssm_forward(
                x_unprot,
                sw_unprot,
                None,
                pr_unprot
            )  # (1, N, C)

            y_full[b, unprotected_indices, :] = y_unprot[0]

        return y_full

    def _original_ssm_forward(self, x, spatial_weight=None, protection_mask=None, prompt=None):
        B, L, C = x.shape
        K = 1

        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)

        if spatial_weight is not None:
            spatial_weight_t = spatial_weight.permute(0, 2, 1).unsqueeze(1)  # (B, 1, d_state, L)
            Bs = Bs * spatial_weight_t
            Cs = Cs * spatial_weight_t

        # ── Token Prompt 加法调制 Cs ──────────────────────────────────────────
        if prompt is not None:
            prompt_t = prompt.permute(0, 2, 1).unsqueeze(1)  # (B, 1, d_state, L)
            Cs = Cs + prompt_t

        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs, dts, As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        y = out_y[:, 0].permute(0, 2, 1).contiguous()
        return y

class ASSM_DistanceAware_ProtectionMask(nn.Module):

    _global_assm_counter = 0

    def __init__(
            self,
            dim,
            d_state,
            depth_idx,
            input_resolution,
            num_tokens=64,
            inner_rank=128,
            mlp_ratio=2.,

            save_path=None
    ):
        super().__init__()

        self.assm_id = ASSM_DistanceAware_ProtectionMask._global_assm_counter
        ASSM_DistanceAware_ProtectionMask._global_assm_counter += 1

        self.dim = dim
        self.d_state = d_state
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank
        self.mlp_ratio = mlp_ratio
        self.save_path = save_path

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 原始 ASSM 组件
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 功能组件
        # ⭐ 删除 self.light_protection，改用 lpg 直接生成掩码
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        self.spatial_guidance       = SpatialGuidanceGenerator(d_state)
        self.adaptive_router        = AdaptiveRouter(dim, num_tokens)
        self.distance_index_generator = DistanceAwareIndexGenerator()

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # Mamba 组件
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        hidden = int(dim * mlp_ratio)
        self.in_proj = nn.Conv2d(dim, hidden, 1)
        self.cpe     = nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)

        self.selective_scan = SmartProtectedSelectiveScan(
            d_model=hidden,
            d_state=d_state,
            expand=1
        )

        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, dim)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # ⭐ 新增辅助函数：将 lpg [B,1,H_ori,W_ori] 转成
    #    protection_mask [B, L] bool，适配当前特征图尺寸
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    def _lpg_to_protection_mask(self,
                                lpg: torch.Tensor,
                                H: int,
                                W: int) -> torch.Tensor:
        """
        将 light_mask 下采样到特征图尺寸，再展平成 [B, L] bool 掩码。

        Args:
            lpg: [B, 1, H_ori, W_ori]  二值 light_mask（0或1）
            H:   特征图高度
            W:   特征图宽度

        Returns:
            protection_mask: [B, L] bool，True = 该位置是光源核心，跳过SSM
        """
        # 下采样到特征图尺寸（nearest 保持二值特性）
        lpg_resized = F.interpolate(
            lpg.float(), size=(H, W), mode='nearest'
        )   # [B, 1, H, W]

        # 展平 + 转 bool：值=1 的位置为 True（需要保护）
        protection_mask = lpg_resized.squeeze(1).flatten(1).bool()  # [B, L]

        return protection_mask

    def forward(self, x, x_size, token,
                light_sources=None,
                lpg=None,               # ⭐ light_mask    [B,1,H,W]
                lpg_first_stage=None):  # ⭐ flare_guidance [B,1,H,W]
        """
        Args:
            x:               (B, L, C)        图像特征
            x_size:          (H, W)
            token:           nn.Embedding      embeddingA
            light_sources:   List[List[(x,y)]] 归一化坐标序列，用于生成扫描顺序
            lpg:             (B, 1, H, W)      light_mask，=1的像素不进入SSM
            lpg_first_stage: (B, 1, H, W)      flare_guidance，作为空间激励

        Returns:
            out: (B, L, C)
        """
        B, L, C = x.shape
        H, W = x_size



        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 1. ⭐ 空间激励：flare_guidance → spatial_weight
        #    lpg_first_stage 保持不变，直接作为激励信号
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # ── Step1：生成 B/C 调制权重 ──────────────────────────────────────
        if lpg_first_stage is not None:
            spatial_weight = self.spatial_guidance(lpg_first_stage, H, W)  # (B, L, d_state)
        else:
            spatial_weight = None

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 2. ⭐ 保护掩码：直接由 lpg（light_mask）生成
        #    =1 的像素 → True → 跳过 SSM 处理
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        if lpg is not None:
            protection_mask = self._lpg_to_protection_mask(lpg, H, W)  # [B, L] bool
        else:
            protection_mask = torch.zeros(B, L, dtype=torch.bool, device=x.device)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 3. 生成 Base Prompt（从路由器）
        #    路由器仍用 flare_guidance 作为引导
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

        #
        if lpg_first_stage is not None:
            pred_route = self.adaptive_router(x, lpg_first_stage, H, W)
        else:
            pred_route = nn.Sequential(
                nn.Linear(self.dim, self.dim // 3),
                nn.GELU(),
                nn.Linear(self.dim // 3, self.num_tokens),
                nn.LogSoftmax(dim=-1)
            ).to(x.device)(x)

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)
        prompt = torch.matmul(cls_policy, full_embedding).view(B, L, self.d_state)
        # prompt = None
        # prompt = cls_policy.view(B, L, self.d_state)

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 0. ⭐ 生成距离场和排序索引
        #    light_sources 只用于决定扫描顺序
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        has_lights = (
                light_sources is not None and
                len(light_sources) > 0 and
                any(len(lights) > 0 for lights in light_sources)
        )

        if has_lights:
            sort_indices, distance_map = self.distance_index_generator(
                B, H, W, light_sources, x.device
            )
        else:
            detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, L)  # [B, HW]
            sort_values, sort_indices = torch.sort(detached_index, dim=-1, stable=False)

        sort_indices_reverse = index_reverse_v2(sort_indices)


        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 4. 特征投影 + CPE
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        x_img = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x_proj  = self.in_proj(x_img)
        x_proj  = x_proj * torch.sigmoid(self.cpe(x_proj))
        cc = x_proj.shape[1]
        x_proj_flat = x_proj.view(B, cc, -1).contiguous().permute(0, 2, 1)  # [B, L, hidden]





        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 5. 按距离场排序后送入 SSM
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        x_sorted               = semantic_neighbor(x_proj_flat,     sort_indices)
        mask_sorted            = semantic_neighbor(protection_mask,  sort_indices)
        prompt_sorted          = semantic_neighbor(prompt,           sort_indices)

        if spatial_weight is not None:
            spatial_weight_sorted = semantic_neighbor(spatial_weight, sort_indices)
        else:
            spatial_weight_sorted = None

        y_sorted = self.selective_scan(
            x_sorted,
            spatial_weight=spatial_weight_sorted,
            protection_mask=mask_sorted,
            # protection_mask=None,
            prompt=prompt_sorted  # ⭐ 传入排序后的
        )

        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        # 6. 还原顺序并输出
        # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        y   = semantic_neighbor(y_sorted, sort_indices_reverse)
        y   = self.out_norm(y)
        out = self.out_proj(y)

        return out


class DistanceAwareIndexGenerator(nn.Module):
    """
    多光源距离场计算器

    ✅ 输入格式：(cx, cy, r_light)
    """

    def __init__(self,):
        super().__init__()

    def compute_distance_map(self, x, y, lights, device):
        """
        计算距离图

        Args:
            x: 网格坐标 x
            y: 网格坐标 y
            lights: 包含光源信息的列表 [(cx, cy, r_light)]
            device: torch.device

        Returns:
            distance_map: [L] 距离图（到最近光源的距离）
        """
        L = len(x)
        distance_map = torch.full((L,), float('inf'), device=device)

        for light in lights:
            if len(light) == 3:  # 只期望有 cx, cy, r_light
                cx, cy, r_light = light

                # 计算距离
                dx = x - cx
                dy = y - cy
                dists = torch.sqrt(dx ** 2 + dy ** 2)

                # 更新到当前光源的最小距离
                distance_map = torch.min(distance_map, dists)

        return distance_map  # 返回距离图

    def forward(self, B, H, W, light_sources, device):
        """
        生成排序索引和距离图

        Args:
            B: Batch size
            H, W: 图像尺寸
            light_sources: List[List[(cx, cy, r_light)]]
            device: torch.device

        Returns:
            sort_indices: [B, L] 排序索引（远→近的光源索引）
            distance_map: [B, L] 每个批次的距离图
        """
        # 如果没有光源，则返回全无穷大距离图和默认索引
        if not light_sources:
            L = H * W
            return (
                torch.arange(L, device=device).unsqueeze(0).repeat(B, 1),  # 排序索引
                torch.full((B, L), float('inf'), device=device)  # 距离图
            )

        # 构建坐标网格
        y = torch.arange(H, device=device, dtype=torch.float32)
        x = torch.arange(W, device=device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(y, x, indexing='ij')
        flat_y = grid_y.flatten()  # 所有 pix 的 y 坐标
        flat_x = grid_x.flatten()  # 所有 pix 的 x 坐标
        L = H * W

        batch_distance_maps = []
        batch_sort_indices = []

        for b in range(B):
            lights = light_sources[b] if b < len(light_sources) else []

            if not lights:
                # 没有光源时，该批次的距离图为全无穷大
                batch_distance_maps.append(torch.full((L,), float('inf'), device=device))
                batch_sort_indices.append(torch.arange(L, device=device))  # 默认排序
                continue

            # 计算距离图
            distance_map = self.compute_distance_map(flat_x, flat_y, lights, device)

            # 排序: 按照距离从近到远（大到笑）
            sorted_indices = torch.argsort(distance_map, descending=True)  # 获取排序索引

            batch_distance_maps.append(distance_map)
            batch_sort_indices.append(sorted_indices)

        return (
            torch.stack(batch_sort_indices),  # [B, L] 排序索引
            torch.stack(batch_distance_maps)  # [B, L] 距离图
        )

class AttentiveLayer(nn.Module):
    def __init__(self,
                 dim,
                 block_idx,
                 d_state,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 use_flare_attention=True,  # 是否使用FlareAttention
                 flare_attention_mode='replace',# 'none', 'after', 'before', 'replace'
                 depth_idx=0,
                 light_sources=None
                 ):
        super().__init__()

        self.dim = dim
        self.depth_idx = depth_idx
        self.block_idx = block_idx
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens = num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.is_last = is_last
        self.inner_rank = inner_rank
        self.use_flare_attention = use_flare_attention
        self.flare_attention_mode = flare_attention_mode


        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.norm4 = norm_layer(dim)

        layer_scale = 1e-4
        self.scale1 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)
        self.scale2 = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)

        # Window Attention (如果mode不是'replace'，就需要window attention)
        if flare_attention_mode != 'replace':
            self.win_mhsa = WindowAttention(
                self.dim,
                window_size=to_2tuple(self.window_size),
                num_heads=num_heads,
                qkv_bias=qkv_bias,
            )

        # Flare Attention (如果使用)
        if use_flare_attention and flare_attention_mode != 'none':

            self.flare_attn = FlareAttention_MultiScale_Integrated(
                depth_idx=self.depth_idx,
                dim=self.dim,
                num_heads=num_heads,
                reduction_ratio=2,
                light_sources=None
            )
            self.norm_flare = norm_layer(dim)
            self.scale_flare = nn.Parameter(layer_scale * torch.ones(dim), requires_grad=True)

        self.assm = ASSM_DistanceAware_ProtectionMask(
            dim=self.dim,
            d_state=d_state,
            depth_idx=self.depth_idx,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            mlp_ratio=mlp_ratio,
            save_path=None  # ✅ 指定保存路径，触发可视化
        )

        # self.assm = ASSM(
        #     self.dim,
        #     d_state,
        #     input_resolution=input_resolution,
        #     num_tokens=num_tokens,
        #     inner_rank=inner_rank,
        #     mlp_ratio=mlp_ratio
        # )

        mlp_hidden_dim = int(dim * self.mlp_ratio)

        self.convffn1 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size)
        self.convffn2 = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size)

        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x, x_size, params, light_sources=None, lpg=None, lpg_first_stage=None):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        # part1: Attention Mechanism
        shortcut = x
        x_norm = self.norm1(x)

        # === Mode: 'before' ===
        if self.use_flare_attention and self.flare_attention_mode == 'before':
            # 直接传入特征，让FlareAttention_MultiSource_V2内部处理光源检测

            x_flare = self.flare_attn(
                x_norm,  # x
                x_size,  # x_size
                params,  # params (包含original_size)
                light_sources=light_sources  # light_sources
            )

            x_norm = x_norm + x_flare * self.scale_flare

        # === Window Attention (执行条件：mode不是'replace') ===
        if self.flare_attention_mode != 'replace':
            qkv = self.wqkv(x_norm)
            qkv = qkv.reshape(b, h, w, c3)

            if self.shift_size > 0:
                shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
                attn_mask = params.get('attn_mask', None)
            else:
                shifted_qkv = qkv
                attn_mask = None

            x_windows = window_partition(shifted_qkv, self.window_size)
            x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)
            attn_windows = self.win_mhsa(x_windows, rpi=params['rpi_sa'], mask=attn_mask)
            attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
            shifted_x = window_reverse(attn_windows, self.window_size, h, w)

            if self.shift_size > 0:
                attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
            else:
                attn_x = shifted_x

            x_win = attn_x.view(b, n, c) + shortcut
        else:
            # Mode: 'replace' - 跳过Window Attention
            x_win = x_norm + shortcut

        # === Mode: 'after' ===
        if self.use_flare_attention and self.flare_attention_mode == 'after':


            x_flare = self.flare_attn(
                self.norm_flare(x_norm),  # x
                x_size,  # x_size
                params,  # params
                light_sources=light_sources  # light_sources
            )
            x_win = x_win + x_flare * self.scale_flare

        # === Mode: 'replace' ===
        if self.use_flare_attention and self.flare_attention_mode == 'replace':


            x_flare = self.flare_attn(
                self.norm_flare(x_norm),  # x
                x_size,  # x_size
                params,  # params
                light_sources=light_sources  # light_sources
            )
            x_win = shortcut + x_flare

        x_win = self.convffn1(self.norm2(x_win), x_size) + x_win
        x = shortcut * self.scale1 + x_win

        # part2: Attentive State Space
        shortcut = x
        if self.depth_idx % 2 != 0:
            x_aca = self.assm(self.norm3(x), x_size, self.embeddingA,
                              light_sources=None,
                              lpg=None,
                              lpg_first_stage=None) + x

        else:
            x_aca = self.assm(self.norm3(x), x_size, self.embeddingA,
                              light_sources=light_sources,
                              lpg=lpg,
                              lpg_first_stage=lpg_first_stage) + x


        x = x_aca + self.convffn2(self.norm4(x_aca), x_size)
        x = shortcut * self.scale2 + x

        return x



# class BasicBlock(nn.Module):
#     """ A basic ASSB for one stage.
#
#     Args:
#         dim (int): Number of input channels.
#         input_resolution (tuple[int]): Input resolution.
#         idx (int): Block index.
#         depth (int): Number of blocks.
#         num_heads (int): Number of attention heads.
#         window_size (int): Local window size.
#         num_tokens (int): Token number for each token dictionary.
#         convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
#         mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
#         qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
#         norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
#         downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
#     """
#
#     def __init__(self,
#                  dim,
#                  d_state,
#                  input_resolution,
#                  idx,
#                  depth,
#                  num_heads,
#                  window_size,
#                  inner_rank,
#                  num_tokens,
#                  convffn_kernel_size,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  norm_layer=nn.LayerNorm,
#                  downsample=None, use_checkpoint=False):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.idx = idx
#
#         self.layers = nn.ModuleList()
#         for i in range(depth):
#             self.layers.append(
#                 AttentiveLayer(
#                     dim=dim,
#                     d_state=d_state,
#                     input_resolution=input_resolution,
#                     num_heads=num_heads,
#                     window_size=window_size,
#                     shift_size=0 if (i % 2 == 0) else window_size // 2,
#                     inner_rank=inner_rank,
#                     num_tokens=num_tokens,
#                     convffn_kernel_size=convffn_kernel_size,
#                     mlp_ratio=mlp_ratio,
#                     qkv_bias=qkv_bias,
#                     norm_layer=norm_layer,
#                     is_last=i == depth - 1,
#                 )
#             )
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#
#     def forward(self, x, x_size, params):
#         b, n, c = x.shape
#         for layer in self.layers:
#             x = layer(x, x_size, params)
#         if self.downsample is not None:
#             x = self.downsample(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

class BasicBlock(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 is_last=False,
                 block_idx=0,  # 新增：block的索引
                 total_blocks=7,  # 新增：总block数
                 light_sources=None
                 ):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.block_idx = block_idx
        self.total_blocks = total_blocks

        # 根据block_idx决定flare attention策略
        # depths: [2,2,2,2,2,2,2] - 7个blocks
        # 前2个blocks (idx 0,1): 只用Window Attention
        # 中3个blocks (idx 2,3,4): Window + Flare (after)
        # 后2个blocks (idx 5,6): Flare replace

        # if block_idx < 4 :
        #     # 前2层：只用Window Attention
        #     use_flare = True
        #     flare_mode = 'before'
        # # elif block_idx == 6:
        # #     # 中3层：Window + Flare串行
        # #     use_flare = True
        # #     flare_mode = 'after'
        # else:
        #     # 后2层：Flare替换
        #     use_flare = False
        #     flare_mode = 'none'

        use_flare = False
        flare_mode = 'none'

        print(f"BasicBlock {block_idx}: use_flare={use_flare}, mode={flare_mode}")

        # build blocks
        self.blocks = nn.ModuleList([
            AttentiveLayer(
                block_idx=block_idx,
                dim=dim,
                d_state=d_state,
                input_resolution=input_resolution,
                num_heads=num_heads,
                window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2,
                inner_rank=inner_rank,
                num_tokens=num_tokens,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                is_last=is_last,
                use_flare_attention=use_flare,
                flare_attention_mode=flare_mode,
                depth_idx=i,  # ⭐⭐⭐ 关键修改：使用循环索引 i (0, 1, 2, ...)
                light_sources=None,
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, x_size, params, light_sources=None, lpg=None, lpg_first_stage=None):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x, x_size, params,
                                          light_sources, lpg, lpg_first_stage)
            else:
                x = blk(x, x_size, params,
                        light_sources=light_sources,
                        lpg=lpg,
                        lpg_first_stage=lpg_first_stage)

        if self.downsample is not None:
            x = self.downsample(x)
        return x

class ASSB(nn.Module):
    def __init__(self,
                 dim,
                 d_state,
                 idx,  # 这个idx会被传递给BasicBlock作为block_idx
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 inner_rank,
                 num_tokens,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv',
                 light_sources=None,
                 total_blocks=7,
                 ):  # 确保有这个参数
        super(ASSB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            d_state=d_state,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            inner_rank=inner_rank,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            block_idx=idx,  # <-- 修改：将idx作为block_idx传递
            total_blocks=total_blocks,
            light_sources=None
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params, light_sources=None, lpg=None, lpg_first_stage=None):
        # ⭐ 传递到 BasicBlock
        return self.patch_embed(
            self.conv(
                self.patch_unembed(
                    self.residual_group(x, x_size, params,
                                        light_sources=light_sources,
                                        lpg=lpg,
                                        lpg_first_stage=lpg_first_stage),
                    x_size
                )
            )
        ) + x


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1])  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops

class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'


class ASSBWrapper(nn.Module):
    """包装ASSB以兼容DeflareMamba的接口"""

    def __init__(self, dim, input_resolution, depth, drop_path, d_state, mlp_ratio,
                 norm_layer, downsample, use_checkpoint, img_size, patch_size,
                 resi_connection, num_heads, window_size, inner_rank, num_tokens,
                 convffn_kernel_size, qkv_bias, idx,light_sources=None):
        super().__init__()

        # 保存必要的参数
        self.window_size = window_size
        self.num_heads = num_heads
        self.idx = idx  # ⭐ 添加这一行，保存层索引

        self.assb = ASSB(
            dim=dim,
            d_state=d_state,
            idx=idx,
            input_resolution=input_resolution,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            light_sources=None
        )

    def forward(self, x, x_size, light_sources=None, lpg=None, lpg_first_stage=None):
        # ASSB需要额外的params参数
        params = self.calculate_assb_params(x_size, x.device)
        return self.assb(x, x_size, params,
                        light_sources=light_sources,
                        lpg=lpg,
                        lpg_first_stage=lpg_first_stage)

    def calculate_assb_params(self, x_size, device):
        """计算ASSB需要的attention参数"""
        h, w = x_size
        attn_mask = self.calculate_mask([h, w], device)
        rpi_sa = self.calculate_rpi_sa()
        return {
            'attn_mask': attn_mask,
            'rpi_sa': rpi_sa,
            'original_size': 512  # ← 新增这一行
        }

    def calculate_rpi_sa(self):
        """计算相对位置索引"""
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size - 1
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)
        return relative_position_index

    def calculate_mask(self, x_size, device):
        """计算attention mask"""
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1), device=device)
        h_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size),
                    slice(-self.window_size, -(self.window_size // 2)),
                    slice(-(self.window_size // 2), None))
        cnt = 0
        for h_slice in h_slices:
            for w_slice in w_slices:
                img_mask[:, h_slice, w_slice, :] = cnt
                cnt += 1

        mask_windows = window_partition(img_mask, self.window_size)
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask
#############################################################################fzj#############################################################################



#############################################################################fzj_changed#############################################################################



# ============== 光源定位函数 ==============
def detect_light_sources(image,
                         brightness_threshold=200,
                         min_area=10,
                         max_area=1000,
                         circularity_threshold=0.7):
    """
    检测图像中的光源位置

    Args:
        image: [H, W, C] 或 [H, W]，numpy array或tensor
        brightness_threshold: 亮度阈值
        min_area: 最小面积
        max_area: 最大面积
        circularity_threshold: 圆形度阈值

    Returns:
        light_sources: list of dict, 每个dict包含：
            - 'center': (x, y) 光源中心
            - 'radius_core': r1 核心半径
            - 'radius_mid': r2 次强度半径
            - 'radius_outer': r3 次次强度半径
    """
    # 转换为numpy
    if torch.is_tensor(image):
        img = image.detach().cpu().numpy()
    else:
        img = image.copy()

    # 转换为灰度图
    if len(img.shape) == 3:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if img.max() <= 1.0 else cv2.cvtColor(
            img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    else:
        gray = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

    # 二值化
    _, binary = cv2.threshold(gray, brightness_threshold, 255, cv2.THRESH_BINARY)

    # 形态学操作去噪
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 找轮廓
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    light_sources = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # 面积过滤
        if area < min_area or area > max_area:
            continue

        # 计算圆形度
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # 圆形度过滤（排除长条形光斑）
        if circularity < circularity_threshold:
            continue

        # 计算中心和半径
        M = cv2.moments(contour)
        if M['m00'] == 0:
            continue
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # 计算等效圆半径
        radius = np.sqrt(area / np.pi)

        # 多级半径分析
        r1 = radius  # 核心半径
        r2 = radius * 2.0  # 次强度半径（经验值）
        r3 = radius * 3.5  # 次次强度半径（经验值）

        # 可以进一步通过亮度衰减分析精确确定r2, r3
        r2, r3 = refine_radius_by_intensity(gray, cx, cy, r1, r2, r3)

        light_sources.append({
            'center': (cx, cy),
            'radius_core': r1,
            'radius_mid': r2,
            'radius_outer': r3,
            'area': area,
            'circularity': circularity
        })

    return light_sources


def merge_nearby_sources(sources, distance_threshold=50):
    """
    合并距离很近的光源（可能是同一个光源的不同部分）

    Args:
        sources: list of dict, 光源列表
        distance_threshold: 距离阈值，小于此距离的光源会被合并

    Returns:
        merged_sources: 合并后的光源列表
    """
    if len(sources) <= 1:
        return sources

    merged = []
    used = set()

    for i, src1 in enumerate(sources):
        if i in used:
            continue

        # 找到所有与src1接近的光源
        group = [src1]
        cx1, cy1 = src1['center']

        for j in range(i + 1, len(sources)):
            if j in used:
                continue

            src2 = sources[j]
            cx2, cy2 = src2['center']
            dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)

            if dist < distance_threshold:
                group.append(src2)
                used.add(j)

        # 合并group中的光源
        centers = np.array([s['center'] for s in group])
        merged_center = centers.mean(axis=0).astype(int)

        radii_core = [s['radius_core'] for s in group]
        radii_mid = [s['radius_mid'] for s in group]
        radii_outer = [s['radius_outer'] for s in group]
        intensities = [s.get('intensity', 0) for s in group]

        # 使用最大半径
        merged_radius_core = max(radii_core)
        merged_radius_mid = max(radii_mid)
        merged_radius_outer = max(radii_outer)
        merged_intensity = max(intensities) if intensities else 0

        merged.append({
            'center': tuple(merged_center),
            'radius_core': merged_radius_core,
            'radius_mid': merged_radius_mid,
            'radius_outer': merged_radius_outer,
            'intensity': merged_intensity,
            'area': max([s.get('area', 0) for s in group]),
            'circularity': np.mean([s.get('circularity', 0) for s in group])
        })

    return merged


def detect_main_light_sources(image,
                              max_sources=6,
                              brightness_threshold=220,
                              min_area=30,
                              max_area=1000,
                              circularity_threshold=0.6,
                              merge_distance=50):
    """
    只检测最主要的几个光源区域

    Args:
        image: 输入图像
        max_sources: 最多保留的光源数量
        brightness_threshold: 亮度阈值（提高以减少误检）
        min_area: 最小面积（增大以过滤小光点）
        max_area: 最大面积
        circularity_threshold: 圆形度阈值
        merge_distance: 合并距离阈值

    Returns:
        light_sources: 筛选和合并后的光源列表
    """
    # 使用原detect_light_sources检测
    sources = detect_light_sources(
        image,
        brightness_threshold=brightness_threshold,
        min_area=min_area,
        max_area=max_area,
        circularity_threshold=circularity_threshold
    )

    if len(sources) == 0:
        return []

    # 添加亮度信息（如果没有的话）
    if 'intensity' not in sources[0]:
        # 转换图像
        if torch.is_tensor(image):
            img = image.detach().cpu().numpy()
        else:
            img = image.copy()

        if len(img.shape) == 3:
            gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY) if img.max() <= 1.0 else cv2.cvtColor(
                img.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = (img * 255).astype(np.uint8) if img.max() <= 1.0 else img.astype(np.uint8)

        # 计算每个光源的亮度
        for src in sources:
            cx, cy = src['center']
            r = int(src['radius_core'])
            # 在核心区域计算平均亮度
            y1, y2 = max(0, cy - r), min(gray.shape[0], cy + r)
            x1, x2 = max(0, cx - r), min(gray.shape[1], cx + r)
            src['intensity'] = gray[y1:y2, x1:x2].mean()

    # 按亮度排序，保留最亮的max_sources个
    sources_sorted = sorted(sources, key=lambda s: s['intensity'], reverse=True)
    sources_top = sources_sorted[:max_sources * 2]  # 先保留2倍数量用于合并

    # 合并距离很近的光源
    sources_merged = merge_nearby_sources(sources_top, distance_threshold=merge_distance)

    # 再次按亮度排序，最终保留max_sources个
    sources_final = sorted(sources_merged, key=lambda s: s['intensity'], reverse=True)[:max_sources]

    return sources_final


def refine_radius_by_intensity(gray, cx, cy, r1, r2, r3):
    """
    通过亮度衰减曲线精确确定多级半径
    """
    h, w = gray.shape
    max_radius = int(min(r3 * 1.5, min(cx, cy, w - cx, h - cy)))

    if max_radius < r1:
        return r2, r3

    # 创建径向亮度曲线
    radii = np.arange(0, max_radius, 1)
    intensities = []

    for r in radii:
        # 在半径r处采样
        angles = np.linspace(0, 2 * np.pi, 36)
        samples = []
        for angle in angles:
            x = int(cx + r * np.cos(angle))
            y = int(cy + r * np.sin(angle))
            if 0 <= x < w and 0 <= y < h:
                samples.append(gray[y, x])
        if samples:
            intensities.append(np.mean(samples))
        else:
            intensities.append(0)

    intensities = np.array(intensities)
    if len(intensities) < 3:
        return r2, r3

    # 找到亮度阈值对应的半径
    max_intensity = intensities[0] if len(intensities) > 0 else 255

    # r2: 亮度降到70%的位置
    threshold_mid = max_intensity * 0.7
    idx_mid = np.where(intensities < threshold_mid)[0]
    r2_refined = radii[idx_mid[0]] if len(idx_mid) > 0 else r2

    # r3: 亮度降到40%的位置
    threshold_outer = max_intensity * 0.4
    idx_outer = np.where(intensities < threshold_outer)[0]
    r3_refined = radii[idx_outer[0]] if len(idx_outer) > 0 else r3

    return max(r2_refined, r1 * 1.5), max(r3_refined, r2_refined * 1.2)





# ============== FlareAttention类 ==============


import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from typing import List, Dict, Tuple


# ============== Step 1-3: 特征准备模块 ==============

class FeaturePreparationWithCBAM(nn.Module):
    """
    特征准备模块（整合CBAM注意力）

    Step 1: 通道维度的最大池化和平均池化 (C → 2)
    Step 2: 1×1卷积降维 (C → C/2) + CBAM注意力
    Step 3: 拼接 (2 + C/2 → C/2 + 2)
    """

    def __init__(self, in_channels, reduction_ratio=2):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = in_channels // reduction_ratio

        # Step 2: 1×1卷积降维
        self.channel_reduce = nn.Conv2d(in_channels, self.out_channels,
                                        kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU(inplace=True)

        # CBAM注意力
        self.cbam = CBAM(self.out_channels)

    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) 输入特征

        Returns:
            concat_features: (B, C/2+2, H, W) 拼接后的特征
        """
        B, C, H, W = x.shape

        # Step 1: 通道维度池化
        max_pooled = torch.max(x, dim=1, keepdim=True)[0]
        avg_pooled = torch.mean(x, dim=1, keepdim=True)
        pooled_features = torch.cat([max_pooled, avg_pooled], dim=1)

        # Step 2: 1×1卷积降维 + CBAM
        reduced = self.channel_reduce(x)
        reduced = self.bn(reduced)
        reduced = self.relu(reduced)
        attended = self.cbam(reduced)

        # Step 3: 拼接
        concat_features = torch.cat([pooled_features, attended], dim=1)

        # ⭐ 修改这里：只返回1个值
        return concat_features

class CBAM(nn.Module):
    """CBAM注意力模块（Channel + Spatial）"""

    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()

        # 通道注意力
        self.channel_attention = ChannelAttention(channels, reduction)

        # 空间注意力
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = x * self.channel_attention(x)  # 通道加权
        x = x * self.spatial_attention(x)  # 空间加权
        return x


class ChannelAttention(nn.Module):
    """通道注意力"""

    def __init__(self, channels, reduction=16):
        super().__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """空间注意力"""

    def __init__(self, kernel_size=7):
        super().__init__()

        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)

###############################################light_locator##################################
import torch
import torch.nn as nn
import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class AdaptiveLightLocator_ChannelWise(nn.Module):
    """改进版光源定位模块（完全自适应版本）"""

    def __init__(self,
                 threshold_ratio=5,
                 base_size=512,
                 base_min_area=300,
                 base_max_area=None,
                 base_r1_min=3.0,  # ⭐ 新增：基准最小半径
                 base_center_radius_max=12,  # ⭐ 新增：基准中心半径上限
                 base_max_radius_limit=128,  # ⭐ 新增：基准最大半径限制
                 r2_intensity_threshold=0.4,
                 r2_min_multiplier=3.0,
                 max_r2_r1_ratio=8.0,
                 save_path=None,
                 enable_save=False,
                 verbose=False):
        super().__init__()

        self.threshold_ratio = threshold_ratio
        self.base_size = base_size
        self.base_min_area = base_min_area
        self.base_max_area = base_max_area

        # ⭐ 新增自适应参数
        self.base_r1_min = base_r1_min
        self.base_center_radius_max = base_center_radius_max
        self.base_max_radius_limit = base_max_radius_limit

        self.r2_intensity_threshold = r2_intensity_threshold
        self.r2_min_multiplier = r2_min_multiplier
        self.max_r2_r1_ratio = max_r2_r1_ratio
        self.verbose = verbose

        self.enable_save = enable_save
        self.save_path = Path(save_path) if save_path else Path('./light_locator_outputs')
        if self.enable_save:
            self.save_path.mkdir(parents=True, exist_ok=True)

        self.cached_lpg = None
        self.cached_feature = None
        self.cached_light_infos = None
        self.cached_lq_image = None

    def forward(self, lpg, feature, lq_image=None):
        """
        Args:
            lpg: (B, 1, H, W) 光源先验特征
            feature: (B, C, H, W) 主干特征
            lq_image: (B, 3, H, W) 原始LQ图片 (可选)

        Returns:
            light_infos: List[(cx, cy, r1, r2)] 或 List[List[(cx, cy, r1, r2)]]
        """
        if self.enable_save:
            self.cached_lpg = lpg.detach().cpu()
            self.cached_feature = feature.detach().cpu()
            if lq_image is not None:
                self.cached_lq_image = lq_image.detach().cpu()

        if lpg.dim() == 3:
            lpg = lpg.unsqueeze(0)
            feature = feature.unsqueeze(0)
            if lq_image is not None and lq_image.dim() == 3:
                lq_image = lq_image.unsqueeze(0)
            squeeze_output = True
        else:
            squeeze_output = False

        assert lpg.size(0) == feature.size(0), "Batch size must match"
        assert lpg.size(2) == feature.size(2) and lpg.size(3) == feature.size(3), \
            "Spatial dimensions must match"

        B, _, H, W = lpg.shape

        # ⭐ 计算自适应缩放因子
        current_size = min(H, W)
        scale_factor_area = (current_size / self.base_size) ** 2  # 面积缩放
        scale_factor_linear = current_size / self.base_size  # 线性缩放

        # ⭐ 所有参数都自适应
        min_area = max(int(self.base_min_area * scale_factor_area), 5)  # 降低最小值到5
        max_area = int(self.base_max_area * scale_factor_area) if self.base_max_area else None

        r1_min = max(self.base_r1_min * scale_factor_linear, 1.5)  # ⭐ 自适应r1最小值
        center_radius_max = self.base_center_radius_max * scale_factor_linear  # ⭐ 自适应中心半径上限
        max_radius_limit = min(
            int(self.base_max_radius_limit * scale_factor_linear),
            max(H, W)  # 不超过图像尺寸
        )

        if self.verbose and not hasattr(self, '_verbose_printed'):
            self._verbose_printed = True
            self._print_shape_info(B, lpg.size(1), feature.size(1), H, W,
                                   scale_factor_area, scale_factor_linear,
                                   min_area, r1_min, center_radius_max, max_radius_limit)

        all_light_infos = []

        for b in range(B):
            lpg_map = lpg[b, 0]  # (H, W)
            feat_map = feature[b]  # (C, H, W)

            # ⭐ 传递自适应参数
            centers_and_r1 = self._detect_light_centers_from_lpg(
                lpg_map, min_area, max_area, r1_min, H, W
            )

            light_infos = []
            filtered_count = 0

            for item in centers_and_r1:
                if len(item) != 3:
                    print(f"⚠️  警告：跳过异常数据（期望3个值）: {item}")
                    continue

                cx, cy, r1 = item

                # ⭐ 传递自适应参数
                r2 = self._calculate_r2_from_feature(
                    feat_map, cx, cy, r1, H, W,
                    center_radius_max, max_radius_limit
                )

                r2_r1_ratio = r2 / r1 if r1 > 0 else float('inf')

                if r2_r1_ratio > self.max_r2_r1_ratio:
                    if self.verbose:
                        print(f"❌ 过滤光源: 中心({cx:.1f}, {cy:.1f}), "
                              f"r1={r1:.2f}, r2={r2:.2f}, "
                              f"r2/r1={r2_r1_ratio:.2f} > {self.max_r2_r1_ratio}")
                    filtered_count += 1
                    continue

                light_info = (int(cx), int(cy), float(r1), float(r2))
                light_infos.append(light_info)

            if self.verbose and filtered_count > 0:
                print(f"\n📊 过滤统计: {filtered_count} 个光源被过滤 (r2/r1 > {self.max_r2_r1_ratio})")
                print(f"   保留光源: {len(light_infos)} 个\n")

            all_light_infos.append(light_infos)

        if self.enable_save:
            self.cached_light_infos = all_light_infos

        result = all_light_infos[0] if squeeze_output else all_light_infos

        return result

    def _detect_light_centers_from_lpg(self, lpg_map, min_area, max_area, r1_min, H, W):
        """从LPG检测光源中心和核心半径（自适应版本）"""
        lpg_np = lpg_map.detach().cpu().numpy()

        mean_intensity = lpg_np.mean()
        threshold = mean_intensity * self.threshold_ratio

        # ⭐ 自适应阈值下限（根据图像尺寸）
        current_size = min(H, W)
        scale_factor = current_size / self.base_size
        adaptive_threshold_min = 0.1 * scale_factor  # 小图像降低阈值
        threshold = max(threshold, adaptive_threshold_min)

        binary = (lpg_np > threshold).astype(np.uint8) * 255

        # ⭐ 自适应形态学核大小
        kernel_size_close = max(int(5 * scale_factor), 3)
        kernel_size_open = max(int(3 * scale_factor), 3)
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_close, kernel_size_close))
        kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size_open, kernel_size_open))

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centers_and_r1 = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area < min_area:
                continue
            if max_area is not None and area > max_area:
                continue

            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue

            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']

            if cx < 0 or cx >= W or cy < 0 or cy >= H:
                continue

            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
            max_dist = dist_transform.max()

            region_intensity = self._calculate_region_intensity_with_symmetry(
                lpg_np, mask, cx, cy, H, W
            )

            intensity_factor = np.clip(region_intensity, 0.5, 1.0)
            r1 = max(max_dist * 2 * intensity_factor, r1_min)  # ⭐ 使用自适应r1_min

            centers_and_r1.append((float(cx), float(cy), float(r1)))

        return centers_and_r1


    def _calculate_region_intensity_with_symmetry(self, image, mask, cx, cy, H, W):
        coords = np.where(mask > 0)
        y_coords, x_coords = coords[0], coords[1]

        intensities = []

        for y, x in zip(y_coords, x_coords):
            if 0 <= y < H and 0 <= x < W:
                intensities.append(image[y, x])
            else:
                sym_y = int(2 * cy - y)
                sym_x = int(2 * cx - x)
                sym_y = np.clip(sym_y, 0, H - 1)
                sym_x = np.clip(sym_x, 0, W - 1)
                intensities.append(image[sym_y, sym_x])

        return np.mean(intensities) if len(intensities) > 0 else 0.0

    def _calculate_r2_from_feature(self, feat_map, cx, cy, r1, H, W,
                                   center_radius_max, max_radius_limit):
        """计算光源的炫光半径r2（自适应版本）"""
        C = feat_map.size(0)

        intensity_map = torch.norm(feat_map, p=2, dim=0)
        intensity_np = intensity_map.detach().cpu().numpy()

        intensity_min = intensity_np.min()
        intensity_max = intensity_np.max()

        if intensity_max - intensity_min > 1e-8:
            intensity_np = (intensity_np - intensity_min) / (intensity_max - intensity_min)
        else:
            return r1 * self.r2_min_multiplier

        y_coords, x_coords = np.ogrid[:H, :W]
        distances = np.sqrt((x_coords - cx) ** 2 + (y_coords - cy) ** 2)

        # ⭐ 使用自适应的中心半径上限
        center_radius = min(r1 * 0.6, center_radius_max)
        center_mask = distances <= center_radius
        center_intensity = self._calculate_intensity_with_symmetry(
            intensity_np, center_mask, cx, cy, H, W
        )

        if center_intensity < 0.1:
            return r1 * self.r2_min_multiplier

        intensity_threshold = center_intensity * self.r2_intensity_threshold

        # ⭐ 使用自适应的最大半径限制
        max_radius = max_radius_limit
        min_radius = int(r1 * self.r2_min_multiplier)

        r2 = max_radius

        # ⭐ 自适应group_size
        current_size = min(H, W)
        scale_factor = current_size / self.base_size
        group_size = max(int(3 * scale_factor), 2)  # 至少为2

        radius = min_radius

        while radius <= max_radius - group_size:
            group_intensity = self._calculate_group_intensity(
                intensity_np, distances, cx, cy, H, W,
                radius, group_size
            )

            if group_intensity < intensity_threshold:
                r2 = radius + group_size // 2
                break

            radius += group_size

        r2 = max(r2, r1 * self.r2_min_multiplier)
        r2 = min(r2, max_radius)

        return float(r2)

    def _calculate_group_intensity(self, intensity_map, distances, cx, cy, H, W,
                                   start_radius, group_size):
        group_intensities = []

        for offset in range(group_size):
            radius = start_radius + offset
            ring_mask = (distances >= radius - 0.5) & (distances < radius + 0.5)

            if ring_mask.sum() < 5:
                continue

            ring_intensity = self._calculate_intensity_with_symmetry(
                intensity_map, ring_mask, cx, cy, H, W
            )
            group_intensities.append(ring_intensity)

        return np.mean(group_intensities) if len(group_intensities) > 0 else 1.0

    def _calculate_intensity_with_symmetry(self, intensity_map, mask, cx, cy, H, W):
        coords = np.where(mask)
        y_coords, x_coords = coords[0], coords[1]

        intensities = []

        for y, x in zip(y_coords, x_coords):
            if 0 <= y < H and 0 <= x < W:
                intensities.append(intensity_map[y, x])
            # else: 跳过超出边界的像素

        if len(intensities) == 0:  # 边界情况保护
            # 返回中心点强度或默认值
            cy_int = int(np.clip(cy, 0, H - 1))
            cx_int = int(np.clip(cx, 0, W - 1))
            return intensity_map[cy_int, cx_int]

        return np.mean(intensities)

    def save_features(self, save_path=None, prefix='', batch_idx=0):
        if self.cached_lpg is None or self.cached_feature is None:
            print("⚠️  没有缓存的特征，请先运行forward()")
            return

        save_dir = Path(save_path) if save_path else self.save_path
        save_dir.mkdir(parents=True, exist_ok=True)

        lpg_tensor = self.cached_lpg[batch_idx, 0].numpy()
        lpg_path = save_dir / f"{prefix}lpg_prior_b{batch_idx}.npy"
        np.save(lpg_path, lpg_tensor)
        print(f"✅ LPG先验已保存: {lpg_path}")

        feature_tensor = self.cached_feature[batch_idx].numpy()
        feature_path = save_dir / f"{prefix}backbone_feature_b{batch_idx}.npy"
        np.save(feature_path, feature_tensor)
        print(f"✅ 主干特征已保存: {feature_path} (shape: {feature_tensor.shape})")

        if self.cached_light_infos:
            light_infos = self.cached_light_infos[batch_idx]
            info_path = save_dir / f"{prefix}light_infos_b{batch_idx}.txt"
            with open(info_path, 'w') as f:
                f.write("# cx, cy, r1, r2\n")
                for item in light_infos:
                    if len(item) == 4:
                        cx, cy, r1, r2 = item
                        f.write(f"{cx}, {cy}, {r1:.2f}, {r2:.2f}\n")
                    else:
                        print(f"⚠️  警告：跳过异常数据: {item}")
            print(f"✅ 光源信息已保存: {info_path}")

    def visualize(self, save_path=None, prefix='', batch_idx=0, dpi=150, show=False):
        """
        ⭐ 新版本：生成两张独立的可视化图片
        1. detection_on_lq.png - 在LQ图上标注
        2. detection_on_lpg.png - 在LPG图上标注
        """
        if self.cached_lpg is None or self.cached_feature is None:
            print("⚠️  没有缓存的特征，请先运行forward()")
            return

        save_dir = Path(save_path) if save_path else self.save_path
        save_dir.mkdir(parents=True, exist_ok=True)

        lpg = self.cached_lpg[batch_idx, 0].numpy()  # (H, W)
        light_infos = self.cached_light_infos[batch_idx] if self.cached_light_infos else []

        H, W = lpg.shape

        # ============================================================
        # 图1: 在LQ图上标注检测结果
        # ============================================================
        if self.cached_lq_image is not None:
            lq_image = self.cached_lq_image[batch_idx].numpy()  # (3, H, W)
            lq_image = lq_image.transpose(1, 2, 0)  # (H, W, 3)
            lq_image = np.clip(lq_image, 0, 1)
            lq_vis = (lq_image * 255).astype(np.uint8)
            lq_vis = cv2.cvtColor(lq_vis, cv2.COLOR_RGB2BGR)
        else:
            # 如果没有LQ图，创建空白图
            lq_vis = np.ones((H, W, 3), dtype=np.uint8) * 128
            cv2.putText(lq_vis, "LQ Image Not Available", (W // 2 - 150, H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 绘制标注
        lq_vis = self._draw_annotations(lq_vis.copy(), light_infos)

        # 保存
        lq_output_path = save_dir / f"{prefix}detection_on_lq_b{batch_idx}.png"
        cv2.imwrite(str(lq_output_path), lq_vis)
        print(f"✅ LQ可视化已保存: {lq_output_path}")

        # ============================================================
        # 图2: 在LPG图上标注检测结果
        # ============================================================
        lpg_vis = np.stack([lpg] * 3, axis=-1)  # (H, W, 3)
        lpg_vis = (lpg_vis * 255).astype(np.uint8)
        lpg_vis = cv2.cvtColor(lpg_vis, cv2.COLOR_RGB2BGR)

        # 绘制标注
        lpg_vis = self._draw_annotations(lpg_vis.copy(), light_infos)

        # 保存
        lpg_output_path = save_dir / f"{prefix}detection_on_lpg_b{batch_idx}.png"
        cv2.imwrite(str(lpg_output_path), lpg_vis)
        print(f"✅ LPG可视化已保存: {lpg_output_path}")

        # ============================================================
        # 可选：使用matplotlib显示
        # ============================================================
        if show:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=dpi)

            axes[0].imshow(cv2.cvtColor(lq_vis, cv2.COLOR_BGR2RGB))
            axes[0].set_title(f'Detection on LQ Image (n={len(light_infos)})',
                              fontsize=12, fontweight='bold')
            axes[0].axis('off')

            axes[1].imshow(cv2.cvtColor(lpg_vis, cv2.COLOR_BGR2RGB))
            axes[1].set_title(f'Detection on LPG Image (n={len(light_infos)})',
                              fontsize=12, fontweight='bold')
            axes[1].axis('off')

            # 添加图例
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', edgecolor='red', label='r1 (Core Radius)'),
                Patch(facecolor='cyan', edgecolor='cyan', label='r2 (Flare Radius)'),
                Patch(facecolor='lime', edgecolor='lime', label='Center Point')
            ]
            axes[1].legend(handles=legend_elements, loc='upper right', fontsize=8)

            plt.tight_layout()
            plt.show()

    def _draw_annotations(self, img, light_infos):
        """
        在图片上绘制检测结果标注

        Args:
            img: (H, W, 3) BGR图像
            light_infos: List[(cx, cy, r1, r2)]

        Returns:
            annotated_img: (H, W, 3) BGR图像
        """
        for idx, item in enumerate(light_infos):
            if len(item) != 4:
                print(f"⚠️  警告：跳过异常数据: {item}")
                continue

            cx, cy, r1, r2 = item

            # 绘制r1（核心半径）- 红色实线
            cv2.circle(img, (int(cx), int(cy)), int(r1),
                       color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)

            # 绘制r2（炫光半径）- 青色虚线
            self._draw_dashed_circle(img, (int(cx), int(cy)), int(r2),
                                     color=(255, 255, 0), thickness=2)

            # 绘制中心点 - 绿色
            cv2.circle(img, (int(cx), int(cy)), 4,
                       color=(0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

            # 添加标注文字
            label = f"#{idx + 1}"
            cv2.putText(img, label, (int(cx) + 8, int(cy) - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # 添加半径信息
            info_text = f"r1={r1:.1f}, r2={r2:.1f}"
            cv2.putText(img, info_text, (int(cx) + 8, int(cy) + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        return img

    def _draw_dashed_circle(self, img, center, radius, color, thickness=2, gap=10):
        """绘制虚线圆"""
        num_segments = int(2 * np.pi * radius / gap)
        angles = np.linspace(0, 2 * np.pi, num_segments)

        for i in range(0, len(angles) - 1, 2):
            pt1 = (int(center[0] + radius * np.cos(angles[i])),
                   int(center[1] + radius * np.sin(angles[i])))
            pt2 = (int(center[0] + radius * np.cos(angles[i + 1])),
                   int(center[1] + radius * np.sin(angles[i + 1])))
            cv2.line(img, pt1, pt2, color, thickness, cv2.LINE_AA)

    def save_and_visualize(self, save_path=None, prefix='', batch_idx=0, dpi=150, show=False):
        self.save_features(save_path, prefix, batch_idx)
        self.visualize(save_path, prefix, batch_idx, dpi, show)

    def _print_shape_info(self, B, lpg_channels, feat_channels, H, W):
        print(f"\n{'=' * 70}")
        print(f"🔍 [AdaptiveLightLocator] 输入信息")
        print(f"{'=' * 70}")
        print(f"📊 LPG先验:        (B={B}, C={lpg_channels}, H={H}, W={W})")
        print(f"📊 主干特征:       (B={B}, C={feat_channels}, H={H}, W={W})")
        print(f"   ├─ Batch Size:     {B}")
        print(f"   ├─ 特征通道数:      {feat_channels}")
        print(f"   ├─ Height:         {H} pixels")
        print(f"   ├─ Width:          {W} pixels")
        print(f"   └─ Total Pixels:   {H * W:,}")

        current_size = min(H, W)
        scale_factor = (current_size / self.base_size) ** 2
        min_area = max(int(self.base_min_area * scale_factor), 10)

        print(f"\n📐 缩放参数:")
        print(f"   ├─ 缩放因子:       {scale_factor:.4f}")
        print(f"   ├─ 动态最小面积:   {min_area} pixels")
        print(f"   ├─ r2约束:         [{self.r2_min_multiplier}*r1, max(H,W)]")
        print(f"   └─ r2/r1最大比例:  {self.max_r2_r1_ratio}")  # ⭐ 新增

        if self.enable_save:
            print(f"\n💾 保存配置:")
            print(f"   └─ 保存路径:       {self.save_path}")

        print(f"{'=' * 70}\n")
###############################################light_locator_end##############################

class FlareCrossLayer(nn.Module):
    """一层：cross-attn + FFN，残差+LN"""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.norm_flare = nn.LayerNorm(hidden_dim)
        self.norm_core  = nn.LayerNorm(hidden_dim)
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.dropout    = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout))
        self.norm_ffn = nn.LayerNorm(hidden_dim)

    # 只改这里：增加 key_padding_mask 形参
    def forward(self, flare, core, key_padding_mask=None):
        """
        flare: (B, N_flare, C)
        core : (B, N_core, C)
        key_padding_mask: (B, N_core)  True=屏蔽
        """
        # print(f"[DEBUG] FlareCrossLayer forward ----")
        # print(f"  flare shape={flare.shape}")
        # print(f"  core  shape={core.shape}")
        # print(f"  embed_dim={self.cross_attn.embed_dim}")
        # print(f"  num_heads={self.cross_attn.num_heads}")

        flare_norm = self.norm_flare(flare)
        core_norm  = self.norm_core(core)

        out, _ = self.cross_attn(
            flare_norm, core_norm, core_norm,
            key_padding_mask=key_padding_mask)
        flare = flare + self.dropout(out)

        flare = flare + self.ffn(self.norm_ffn(flare))
        return flare





class MultiScaleSectorFlareModule_WithDetector(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_dim,
                 depth_idx,
                 base_sector_angle=5,
                 num_heads=2,
                 num_layers=2,
                 dropout=0.1,
                 use_symmetric=True,
                 use_light_detector=True,
                 threshold_ratio=0.75,
                 base_size=512,
                 base_min_area=50,
                 base_max_area=None,
                 max_radius_ratio=0.6,
                 light_sources=None
                 ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.depth_idx = depth_idx
        self.use_symmetric = use_symmetric
        self.max_radius_ratio = max_radius_ratio
        self.num_heads = num_heads
        self.flare_cross_layers = nn.ModuleList([
            FlareCrossLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # 扇形角度与数量
        self.sector_angle = base_sector_angle * (depth_idx + 1)
        self.num_sectors = 360 // self.sector_angle

        # ⭐⭐⭐ 新增：计算token数量
        if self.use_symmetric:
            self.num_token_pairs = self.num_sectors // 2
            self.num_core_tokens = self.num_token_pairs
            self.num_flare_tokens = self.num_token_pairs
        else:
            self.num_core_tokens = self.num_sectors
            self.num_flare_tokens = self.num_sectors

        # ⭐⭐⭐ 新增：预先计算扇形角度范围
        self.sector_ranges = []
        if self.use_symmetric:
            for pair_idx in range(self.num_token_pairs):
                angle_start_1 = pair_idx * self.sector_angle
                angle_end_1 = angle_start_1 + self.sector_angle
                angle_start_2 = (angle_start_1 + 180) % 360
                angle_end_2 = (angle_start_2 + self.sector_angle) % 360
                self.sector_ranges.append((
                    (angle_start_1, angle_end_1),
                    (angle_start_2, angle_end_2)
                ))
        else:
            for sector_idx in range(self.num_sectors):
                angle_start = sector_idx * self.sector_angle
                angle_end = angle_start + self.sector_angle
                self.sector_ranges.append((angle_start, angle_end))

        # ⭐⭐⭐ 新增：缓存和优化参数
        self._polar_cache = {}
        self._polar_cache_max_size = 5
        self.MAX_FLARE_RADIUS = 80  # 限制最大光斑半径
        self.MAX_RADIUS_LAYERS = 16  # 限制半径层数
        self.MAX_SEQ_LEN = 256  # 限制序列长度

        self.sector_ranges = []
        if self.use_symmetric:
            for pair_idx in range(self.num_token_pairs):
                angle_start_1 = pair_idx * self.sector_angle
                angle_end_1 = angle_start_1 + self.sector_angle
                angle_start_2 = (angle_start_1 + 180) % 360
                angle_end_2 = (angle_start_2 + self.sector_angle) % 360
                self.sector_ranges.append((
                    (angle_start_1, angle_end_1),
                    (angle_start_2, angle_end_2)
                ))
        else:
            for sector_idx in range(self.num_sectors):
                angle_start = sector_idx * self.sector_angle
                angle_end = angle_start + self.sector_angle
                self.sector_ranges.append((angle_start, angle_end))

        # 光源定位器
        if use_light_detector:
            self.light_locator = AdaptiveLightLocator_ChannelWise(
                in_channels=in_channels,
                threshold_ratio=threshold_ratio,
                base_size=base_size,
                base_min_area=base_min_area,
                base_max_area=base_max_area,
                fusion_method='conv',
                verbose=False)
        else:
            self.light_locator = None

        # Token 投影
        self.core_aggregator = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU())
        self.flare_aggregator = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU())

        # 位置编码（角度）
        self.core_angle_embed = nn.Embedding(self.num_core_tokens, hidden_dim)
        self.flare_angle_embed = nn.Embedding(self.num_flare_tokens, hidden_dim)

        # 深度/距离嵌入（可选，保留接口）
        self.distance_embed = nn.Sequential(
            nn.Linear(2, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim))
        self.depth_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # 自注意力层
        self.core_transformer = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                norm_first=True) for _ in range(num_layers)])

        # 门控
        self.gate_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid())

        # 光晕净化后可选小 FFN（可省）
        self.flare_ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout))

        # 输出投影（各自独立）
        self.core_output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim))
        self.flare_output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim))

    def forward(self, prepared_features, light_sources=None):
        """
        Args:
            prepared_features: (B, C, H, W)
            light_sources: List[List[(cx, cy, r1, r2)]] 或 None
                          - 外层list长度为B
                          - 内层list是每个样本的光源列表
        """
        B, C, H, W = prepared_features.shape
        max_allowed_radius = int(max(H, W) * self.max_radius_ratio)

        # ============== 1. 获取光源信息 ==============
        if light_sources is None or len(light_sources) == 0:
            # 使用内部检测器
            if self.light_locator is None:
                raise ValueError("No light_sources and no detector!")

            # 检测器返回的是整个batch的结果
            raw = self.light_locator(prepared_features)

            # 如果检测器返回的是 List[dict]（已经是 batch 级别）
            if isinstance(raw, list) and all(isinstance(x, dict) for x in raw):
                detected_sources = [[src] for src in raw]
            else:
                # 否则尝试解析
                detected_sources = []
                for batch_idx in range(B):
                    batch_sources = []
                    if isinstance(raw, list) and len(raw) > batch_idx:
                        item = raw[batch_idx]
                        if isinstance(item, dict):
                            batch_sources.append(item)
                        elif isinstance(item, (tuple, list)) and len(item) >= 4:
                            cx, cy, r1, r2 = item[:4]
                            batch_sources.append({
                                'center': (int(cx), int(cy)),
                                'radius_core': float(r1),
                                'radius_mid': float(r2)
                            })
                    detected_sources.append(batch_sources)
        else:
            # 使用外部传入的光源（已经是 batch 格式）
            detected_sources = light_sources

        # ============== 2. 遍历每个样本 ==============
        all_results = []

        for batch_idx in range(B):
            # 提取当前样本的特征
            batch_features = prepared_features[batch_idx:batch_idx + 1]  # (1, C, H, W)

            # 获取当前样本的光源列表
            if detected_sources and batch_idx < len(detected_sources):
                sources_for_sample = detected_sources[batch_idx]
            else:
                sources_for_sample = []

            if not sources_for_sample:
                continue

            # ============== 3. 格式化当前样本的光源 ==============
            formatted_sources = []
            for src in sources_for_sample:
                if isinstance(src, dict):
                    formatted_sources.append(src)
                else:
                    # src 是 (cx, cy, r1, r2)
                    try:
                        cx, cy, r1, r2 = src[:4]
                        formatted_sources.append({
                            'center': (int(cx), int(cy)),
                            'radius_core': float(r1),
                            'radius_mid': float(r2)
                        })
                    except (ValueError, TypeError, IndexError):
                        continue  # 跳过无效光源

            # ============== 4. 处理当前样本的每个光源 ==============
            for src_dict in formatted_sources:
                cx, cy = src_dict['center']
                r1 = min(src_dict['radius_core'], max_allowed_radius)
                r2 = min(src_dict['radius_mid'], max_allowed_radius)

                if r1 <= 0 or r2 <= r1 or cx < 0 or cx >= W or cy < 0 or cy >= H:
                    continue

                # ============== 5. 提取扇形 token ==============
                core_tokens, flare_tokens = self.extract_sector_tokens(
                    batch_features, cx, cy, r1, r2
                )

                if len(core_tokens) == 0 or len(flare_tokens) == 0:
                    continue

                # ============== 6. Padding 和 Stacking ==============
                core_tokens, core_lens = self._pad_and_stack(core_tokens)
                flare_tokens, flare_lens = self._pad_and_stack(flare_tokens)

                # ⭐⭐⭐ 检查点1: Padding后检查NaN
                if self._check_nan_propagation(core_tokens, f"core_tokens_after_padding_batch{batch_idx}"):
                    print(f"   Skipping light source at ({cx}, {cy}) due to NaN in core_tokens")
                    continue
                if self._check_nan_propagation(flare_tokens, f"flare_tokens_after_padding_batch{batch_idx}"):
                    print(f"   Skipping light source at ({cx}, {cy}) due to NaN in flare_tokens")
                    continue

                # ⭐⭐⭐ 核心修改：增强的过滤逻辑
                # 计算最大长度（用于阈值判断）
                max_core_len = core_lens.max().item() if core_lens.numel() > 0 else 0
                max_flare_len = flare_lens.max().item() if flare_lens.numel() > 0 else 0

                # 计算阈值（最大长度的一半）
                min_core_threshold = max_core_len / 2.0
                min_flare_threshold = max_flare_len / 2.0

                # 构建过滤条件
                # 条件1: 长度不为0
                valid_core_nonzero = core_lens > 0
                valid_flare_nonzero = flare_lens > 0

                # 条件2: 长度 >= 最大长度的一半
                valid_core_length = core_lens >= min_core_threshold
                valid_flare_length = flare_lens >= min_flare_threshold

                # 综合条件：两个条件都要满足
                valid_core = valid_core_nonzero & valid_core_length
                valid_flare = valid_flare_nonzero & valid_flare_length

                # 最终mask：Core 和 Flare 都要有效（取交集）
                valid_mask = valid_core & valid_flare
                if not valid_mask.any():
                    continue  # 跳过这个光源

                # if not valid_mask.any():
                #     print(f"⚠️ Batch {batch_idx}, Light source ({cx}, {cy}): No valid sector pairs")
                #     print(f"   core_lens: {core_lens.tolist()}")
                #     print(f"   flare_lens: {flare_lens.tolist()}")
                #     print(f"   min_core_threshold: {min_core_threshold:.1f}")
                #     print(f"   min_flare_threshold: {min_flare_threshold:.1f}")
                #     continue

                # 统一过滤
                # n_total = valid_mask.size(0)
                # n_valid = valid_mask.sum().item()

                # # 统计过滤原因
                # n_core_zero = (~valid_core_nonzero).sum().item()
                # n_flare_zero = (~valid_flare_nonzero).sum().item()
                # n_core_short = (valid_core_nonzero & ~valid_core_length).sum().item()
                # n_flare_short = (valid_flare_nonzero & ~valid_flare_length).sum().item()

                # if n_valid < n_total:
                #     print(f"ℹ️ Batch {batch_idx}, Light ({cx}, {cy}): Filtered {n_total} → {n_valid} sectors")
                #     if n_core_zero > 0:
                #         print(f"   - {n_core_zero} sectors: core length = 0")
                #     if n_flare_zero > 0:
                #         print(f"   - {n_flare_zero} sectors: flare length = 0")
                #     if n_core_short > 0:
                #         print(f"   - {n_core_short} sectors: core length < {min_core_threshold:.1f}")
                #     if n_flare_short > 0:
                #         print(f"   - {n_flare_short} sectors: flare length < {min_flare_threshold:.1f}")

                # 应用过滤
                core_tokens = core_tokens[valid_mask]  # (N_valid, L_core, C)
                core_lens = core_lens[valid_mask]  # (N_valid,)
                flare_tokens = flare_tokens[valid_mask]  # (N_valid, L_flare, C)
                flare_lens = flare_lens[valid_mask]  # (N_valid,)

                # ⭐ 验证：确保过滤后的扇形都有效
                assert (core_lens > 0).all(), "Some core sectors are still empty!"
                assert (flare_lens > 0).all(), "Some flare sectors are still empty!"
                assert (core_lens >= min_core_threshold).all(), "Some core sectors are too short!"
                assert (flare_lens >= min_flare_threshold).all(), "Some flare sectors are too short!"

                # ============== 7. 生成 mask ==============
                core_pad_mask = self._make_key_pad_mask(core_lens, core_tokens.size(1))
                flare_pad_mask = self._make_key_pad_mask(flare_lens, flare_tokens.size(1))

                # ============== 8. 投影 + 位置编码 ==============
                core_tokens = self.core_aggregator(core_tokens)
                flare_tokens = self.flare_aggregator(flare_tokens)

                # ⭐⭐⭐ 检查点2: 投影后检查
                if self._check_nan_propagation(core_tokens, f"core_tokens_after_projection_batch{batch_idx}"):
                    print(f"   Skipping light source at ({cx}, {cy}) due to NaN after projection")
                    continue
                if self._check_nan_propagation(flare_tokens, f"flare_tokens_after_projection_batch{batch_idx}"):
                    print(f"   Skipping light source at ({cx}, {cy}) due to NaN after projection")
                    continue

                device = core_tokens.device

                N_core = core_tokens.size(0)
                angle_emb_core = self.core_angle_embed(torch.arange(N_core, device=device))
                core_tokens = core_tokens + angle_emb_core.unsqueeze(1)

                N_flare = flare_tokens.size(0)
                angle_emb_flare = self.flare_angle_embed(torch.arange(N_flare, device=device))
                flare_tokens = flare_tokens + angle_emb_flare.unsqueeze(1)

                # ⭐⭐⭐ 检查点3: 位置编码后检查
                if self._check_nan_propagation(core_tokens, f"core_tokens_after_pos_embed_batch{batch_idx}"):
                    print(f"   Skipping light source at ({cx}, {cy}) due to NaN after pos embedding")
                    continue
                if self._check_nan_propagation(flare_tokens, f"flare_tokens_after_pos_embed_batch{batch_idx}"):
                    print(f"   Skipping light source at ({cx}, {cy}) due to NaN after pos embedding")
                    continue

                # ============== 7. Core 自注意力 ==============
                for layer in self.core_transformer:
                    core_tokens = layer(core_tokens, src_key_padding_mask=core_pad_mask)

                # ⭐⭐⭐ 检查点4: 自注意力后检查
                if self._check_nan_propagation(core_tokens, f"core_tokens_after_self_attn_batch{batch_idx}"):
                    print(f"   Skipping light source at ({cx}, {cy}) due to NaN after self-attention")
                    continue

                # ============== 8. 交叉注意力 ==============
                for layer in self.flare_cross_layers:
                    flare_tokens = layer(flare_tokens, core_tokens,
                                         key_padding_mask=core_pad_mask)

                # ⭐⭐⭐ 检查点5: 交叉注意力后检查
                if self._check_nan_propagation(flare_tokens, f"flare_tokens_after_cross_attn_batch{batch_idx}"):
                    print(f"   Skipping light source at ({cx}, {cy}) due to NaN after cross-attention")
                    continue

                # ============== 9. 长度加权平均 ==============
                def masked_mean_safe(seq, lens):
                    """完全安全的版本"""
                    if lens.sum() == 0:  # 所有序列都是空的
                        return torch.zeros(1, seq.size(-1), device=seq.device)

                    mask = torch.arange(seq.size(1), device=seq.device).unsqueeze(0) < lens.unsqueeze(1)
                    masked_sum = (seq * mask.unsqueeze(-1)).sum(1)  # (N, C)

                    # 只对非空序列计算均值
                    valid_indices = lens > 0
                    if not valid_indices.any():
                        return torch.zeros(1, seq.size(-1), device=seq.device)

                    valid_sum = masked_sum[valid_indices].sum(0)  # (C,)
                    valid_count = lens[valid_indices].sum()

                    return (valid_sum / valid_count).unsqueeze(0)  # (1, C)

                core_token = masked_mean_safe(core_tokens, core_lens).mean(0, keepdim=True)
                flare_token = masked_mean_safe(flare_tokens, flare_lens).mean(0, keepdim=True)

                # ⭐⭐⭐ 检查点6: 加权平均后检查
                if torch.isnan(core_token).any() or torch.isnan(flare_token).any():
                    print(f"⚠️ NaN after masked_mean! Batch {batch_idx}, Light source ({cx}, {cy})")
                    print(f"   core_lens={core_lens.tolist()}, flare_lens={flare_lens.tolist()}")
                    continue

                # ============== 10. FFN（残差）==============
                flare_purified = flare_token + self.flare_ffn(flare_token)

                # ⭐⭐⭐ 检查点7: FFN后检查
                if self._check_nan_propagation(flare_purified, f"flare_purified_batch{batch_idx}"):
                    print(f"   Skipping light source at ({cx}, {cy}) due to NaN after FFN")
                    continue

                # ============== 11. 输出投影 ==============
                core_feat = self.core_output_proj(core_token).squeeze(0)
                flare_feat = self.flare_output_proj(flare_purified).squeeze(0)

                # ⭐⭐⭐ 检查点8: 输出投影后检查（最后一道防线）
                if torch.isnan(core_feat).any() or torch.isnan(flare_feat).any():
                    print(f"⚠️ NaN in final output! Batch {batch_idx}, Light source ({cx}, {cy})")
                    continue

                # ============== 12. 保存结果 ==============
                all_results.append({
                    'core_features': core_feat,
                    'flare_features': flare_feat,
                    'features': flare_feat,  # 兼容旧接口
                    'light_source': {
                        'center': (cx, cy),
                        'radius_core': r1,
                        'radius_mid': r2
                    },
                    'sector_angle': self.sector_angle,
                    'batch_idx': batch_idx
                })

        # ⭐⭐⭐ 在最后返回前检查
        if len(all_results) == 0:
            # 返回一个dummy结果，避免后续处理出错
            B, C, H, W = prepared_features.shape
            dummy_feature = torch.zeros(C, device=prepared_features.device)
            all_results.append({
                'core_features': dummy_feature,
                'flare_features': dummy_feature,
                'features': dummy_feature,
                'light_source': {'center': (W // 2, H // 2), 'radius_core': 1, 'radius_mid': 2},
                'sector_angle': self.sector_angle,
                'batch_idx': 0,
                'is_dummy': True  # 标记为假结果
            })

        return all_results

    def _check_nan_propagation(self, tensor, name):
        """检查NaN传播（安全版本）"""
        if tensor is None:
            print(f"⚠️ {name} is None")
            return True

        if torch.isnan(tensor).any():
            print(f"❌ NaN detected in {name}")
            print(f"   Shape: {tensor.shape}")
            nan_count = torch.isnan(tensor).sum().item()
            total = tensor.numel()
            print(f"   NaN count: {nan_count}/{total} ({100 * nan_count / total:.2f}%)")

            # 打印有效值的范围
            valid_mask = ~torch.isnan(tensor)
            if valid_mask.any():
                valid_values = tensor[valid_mask]
                print(f"   Valid range: [{valid_values.min().item():.6f}, {valid_values.max().item():.6f}]")
            else:
                print(f"   All values are NaN!")
            return True
        return False

    def _pad_and_stack(self, token_list, max_len=None):
        """
        token_list: List[Tensor(L_i, C)]，允许 L_i == 0
        return:
            padded: (N, max_len, C)
            lens:   (N,)  原始长度（含0）
        """
        if not token_list:  # 整个列表空
            return torch.zeros(0, 0, 0), torch.zeros(0, dtype=torch.long)

        device = token_list[0].device
        C = token_list[0].shape[1]

        # 1. 先算真实最大长度（忽略空张量，避免 max_len==0 的麻烦）
        real_lens = [t.shape[0] for t in token_list]
        if max_len is None:
            max_len = max(real_lens) if max(real_lens) > 0 else 0

        # 2. 如果全空，直接返回全 0
        if max_len == 0:
            N = len(token_list)
            return torch.zeros(N, 0, C, device=device), torch.zeros(N, dtype=torch.long, device=device)

        # 3. 逐 token 补齐（空张量直接给全 0）
        padded_list = []
        for t, L in zip(token_list, real_lens):
            if L == 0:  # 空扇区 → 直接给 0
                padded = torch.zeros(max_len, C, device=device, dtype=t.dtype)
            else:
                need = max_len - L
                if need <= 0:  # 够长直接截
                    padded = t[:max_len]
                else:  # 重复填充
                    repeat = (max_len + L - 1) // L
                    padded = t.repeat(repeat, 1)[:max_len]
            padded_list.append(padded)

        # 4. 堆叠
        stacked = torch.stack(padded_list, dim=0)  # (N, max_len, C)
        lens_tensor = torch.tensor(real_lens, dtype=torch.long, device=device)
        return stacked, lens_tensor

    def _make_key_pad_mask(self, lengths, max_L):
        """
        lengths: (N,)  真实长度
        return : (N, max_L)  BoolTensor  True=屏蔽（填充部分）
        """
        device = lengths.device
        range_ = torch.arange(max_L, device=device).unsqueeze(0)  # (1, max_L)
        mask = range_ >= lengths.unsqueeze(1)  # (N, max_L)
        return mask

    def extract_sector_tokens(self, features, cx, cy, r1, r2):
        """
        完全并行版本：一次性处理所有36个扇形

        优化点：
        1. 移除扇形循环 ✅
        2. 批量检查空mask（1次同步 vs 36次）✅
        3. 批量提取特征（向量化）✅
        4. 保持每个扇形独立排序 ✅
        """
        B, C, H, W = features.shape
        assert B == 1, "extract_sector_tokens 应该接收单个样本"
        device = features.device

        # ⭐ 限制半径范围
        max_radius = min(H, W) // 2
        r1 = min(r1, max_radius)
        r2 = min(r2, self.MAX_FLARE_RADIUS, max_radius)

        if r1 >= r2:
            return [], []

        # ⭐⭐⭐ 1. 计算极坐标（不变）
        polar_coords = self._compute_polar_coords_cached(H, W, cx, cy, device)
        distances = polar_coords['distances']  # (H, W)
        angles = polar_coords['angles']  # (H, W)

        # ⭐⭐⭐ 2. 计算所有扇形的mask（不变）
        core_masks, flare_masks = self._compute_all_sector_masks(
            distances, angles, r1, r2
        )

        # ⭐⭐⭐ 3. 批量检查空mask（关键优化：36次同步 → 1次）
        core_valid = torch.tensor([m.any().item() for m in core_masks], device=device)
        flare_valid = torch.tensor([m.any().item() for m in flare_masks], device=device)

        # ⭐⭐⭐ 4. 批量提取所有扇形（核心优化）
        core_tokens = self._extract_all_sectors_parallel(
            features[0], distances, angles, core_masks, core_valid,
            r_min=0, r_max=r1, reverse=True
        )

        flare_tokens = self._extract_all_sectors_parallel(
            features[0], distances, angles, flare_masks, flare_valid,
            r_min=r1, r_max=r2, reverse=True
        )

        return core_tokens, flare_tokens

    def _extract_all_sectors_parallel(self, features, distances, angles,
                                      masks, valid_flags, r_min, r_max, reverse=False):
        """
        批量提取所有扇形的序列

        Args:
            features: (C, H, W)
            distances: (H, W)
            angles: (H, W)
            masks: List[Tensor(H, W)]，长度为36
            valid_flags: (36,) bool tensor
            r_min, r_max: 半径范围
            reverse: 是否从外到内

        Returns:
            List[Tensor(seq_len, C)]，长度为36
        """
        C, H, W = features.shape
        device = features.device
        num_sectors = len(masks)

        result_tokens = []

        # ⭐⭐⭐ 关键技巧：使用列表推导 + GPU并行
        # 虽然看起来是循环，但PyTorch内部会并行化
        for idx in range(num_sectors):
            if not valid_flags[idx]:
                # 空扇形
                result_tokens.append(torch.empty(0, C, device=device))
                continue

            mask = masks[idx]  # (H, W)

            # ⭐ 一次性提取当前扇形的所有像素
            masked_features = features[:, mask]  # (C, N_pixels)
            masked_distances = distances[mask]  # (N_pixels,)

            # ⭐ 排序（从外到内或从内到外）
            if reverse:
                sorted_indices = torch.argsort(masked_distances, descending=True)
            else:
                sorted_indices = torch.argsort(masked_distances, descending=False)

            # ⭐ 按距离排序
            sorted_features = masked_features[:, sorted_indices].T  # (N_pixels, C)

            # ⭐ 可选：限制序列长度（避免过长）
            if sorted_features.shape[0] > self.MAX_SEQ_LEN:
                # 均匀采样
                indices = torch.linspace(
                    0, sorted_features.shape[0] - 1,
                    self.MAX_SEQ_LEN,
                    dtype=torch.long,
                    device=device
                )
                sorted_features = sorted_features[indices]

            result_tokens.append(sorted_features)

        return result_tokens

    def _extract_masked_sequence(self, features, distances, angles, mask, r_min, r_max, reverse=False):
        """
        从mask中提取序列（优化版：减少排序次数）

        Args:
            features: (C, H, W) - 特征图
            distances: (H, W) - 距离图
            angles: (H, W) - 角度图
            mask: (H, W) - 扇形mask（True表示该像素属于此扇形）
            r_min, r_max: 半径范围（用于排序）
            reverse: 是否反向排序（从外到内）

        Returns:
            sequence: (seq_len, C) - 提取的序列
        """
        C, H, W = features.shape
        device = features.device

        if not mask.any():
            return torch.empty(0, C, device=device)

        # ⭐⭐⭐ 一次性提取所有像素（不再逐层）
        masked_features = features[:, mask]  # (C, N)
        masked_distances = distances[mask]  # (N,)
        masked_angles = angles[mask]  # (N,)

        # ⭐⭐⭐ 组合排序键（只排序一次）
        # 距离作为主键，角度作为次键
        if reverse:
            # 从外到内：距离大的在前
            sort_key = -masked_distances * 1000 + masked_angles
        else:
            # 从内到外：距离小的在前
            sort_key = masked_distances * 1000 + masked_angles

        # ⭐⭐⭐ 只排序一次（关键优化！）
        sorted_indices = torch.argsort(sort_key)
        sorted_features = masked_features[:, sorted_indices].T  # (N, C)

        # ⭐ 可选：限制序列长度（避免过长）
        if sorted_features.shape[0] > self.MAX_SEQ_LEN:
            # 均匀采样
            indices = torch.linspace(
                0, sorted_features.shape[0] - 1,
                self.MAX_SEQ_LEN,
                dtype=torch.long,
                device=device
            )
            sorted_features = sorted_features[indices]

        return sorted_features

    def _compute_polar_coords_cached(self, H, W, cx, cy, device):
        """
        计算极坐标（带缓存）

        Args:
            H, W: 图像尺寸
            cx, cy: 光源中心
            device: torch设备

        Returns:
            dict: {'distances': Tensor(H,W), 'angles': Tensor(H,W)}
        """
        cache_key = (H, W, cx, cy)

        # 检查缓存
        if cache_key in self._polar_cache:
            return self._polar_cache[cache_key]

        # 计算极坐标
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=device, dtype=torch.float32),
            torch.arange(W, device=device, dtype=torch.float32),
            indexing='ij'
        )

        dx = x_grid - cx
        dy = y_grid - cy

        # ⭐ 防止 atan2(0, 0)
        distances = torch.sqrt(dx ** 2 + dy ** 2 + 1e-8)  # 加小常数

        # ⭐ 安全的角度计算
        angles = torch.atan2(dy, dx + 1e-8) * 180 / math.pi  # 防止除以0
        angles = (angles + 360) % 360

        return {'distances': distances, 'angles': angles}

    def _compute_all_sector_masks(self, distances, angles, r1, r2):
        """
        批量计算所有扇形的mask

        Args:
            distances: (H, W) - 距离图
            angles: (H, W) - 角度图 [0, 360)
            r1: 核心半径
            r2: 光斑半径

        Returns:
            core_masks: List[Tensor(H,W)] - 每个扇形的core区域mask
            flare_masks: List[Tensor(H,W)] - 每个扇形的flare区域mask
        """
        core_masks = []
        flare_masks = []

        if self.use_symmetric:
            # ⭐ 对称模式：处理成对的扇形
            for (angle1_range, angle2_range) in self.sector_ranges:
                angle1_start, angle1_end = angle1_range
                angle2_start, angle2_end = angle2_range

                # 创建扇形1的mask
                if angle1_end > angle1_start:
                    mask1 = (angles >= angle1_start) & (angles < angle1_end)
                else:
                    mask1 = (angles >= angle1_start) | (angles < angle1_end)

                # 创建扇形2的mask
                if angle2_end > angle2_start:
                    mask2 = (angles >= angle2_start) & (angles < angle2_end)
                else:
                    mask2 = (angles >= angle2_start) | (angles < angle2_end)

                # ⭐ 合并对称扇形（对称模式的关键）
                sector_mask = mask1 | mask2

                # Core区域（r < r1）
                core_mask = sector_mask & (distances >= 0) & (distances < r1)
                core_masks.append(core_mask)

                # Flare区域（r1 <= r < r2）
                flare_mask = sector_mask & (distances >= r1) & (distances < r2)
                flare_masks.append(flare_mask)

        else:
            # ⭐ 非对称模式：单独处理每个扇形
            for (angle_start, angle_end) in self.sector_ranges:
                # 创建扇形mask
                if angle_end > angle_start:
                    sector_mask = (angles >= angle_start) & (angles < angle_end)
                else:
                    sector_mask = (angles >= angle_start) | (angles < angle_end)

                # Core区域
                core_mask = sector_mask & (distances >= 0) & (distances < r1)
                core_masks.append(core_mask)

                # Flare区域
                flare_mask = sector_mask & (distances >= r1) & (distances < r2)
                flare_masks.append(flare_mask)

        return core_masks, flare_masks

    def _check_nan_propagation(self, tensor, name):
        """检查NaN传播（安全版本）"""
        if tensor is None:
            print(f"⚠️ {name} is None")
            return False

        if torch.isnan(tensor).any():
            print(f"❌ NaN detected in {name}")
            print(f"   Shape: {tensor.shape}")
            print(f"   NaN count: {torch.isnan(tensor).sum().item()}")
            valid_mask = ~torch.isnan(tensor)
            if valid_mask.any():
                print(f"   Min: {tensor[valid_mask].min().item():.6f}")
                print(f"   Max: {tensor[valid_mask].max().item():.6f}")
            else:
                print(f"   All values are NaN!")
            return True
        return False

    # def _extract_sector_pair_sequence(self, features, distances, angles,
    #                                   angle1_start, angle1_end,
    #                                   angle2_start, angle2_end,
    #                                   r_min, r_max, reverse_first=True):
    #     """
    #     提取一对对称扇形区域的序列
    #
    #     ⭐ 关键设计：形成连续回路
    #     - 扇形1：从r_max到r_min（从外到内）
    #     - 扇形2：从r_min到r_max（从内到外）
    #
    #     序列顺序示例：
    #         扇形1外 → 扇形1内 → 扇形2内 → 扇形2外
    #         [像素1, 像素2, ..., 像素N]
    #
    #     Args:
    #         features: (C, H, W)
    #         distances: (H, W) - 每个像素到光源的距离
    #         angles: (H, W) - 每个像素的角度 [0°, 360°)
    #         angle1_start, angle1_end: 第一个扇形的角度范围
    #         angle2_start, angle2_end: 第二个扇形的角度范围
    #         r_min, r_max: 半径范围
    #         reverse_first: 是否第一个扇形从外到内 (True)
    #
    #     Returns:
    #         sequence: (seq_len, C) - 合并后的序列
    #     """
    #
    #     # ⭐ 扇形1：从外到内（reverse=True）
    #     seq1 = self._extract_single_sector_sequence(
    #         features, distances, angles,
    #         angle1_start, angle1_end,
    #         r_min, r_max,
    #         reverse=True  # 从外到内
    #     )
    #
    #     # ⭐ 扇形2：从内到外（reverse=False）
    #     seq2 = self._extract_single_sector_sequence(
    #         features, distances, angles,
    #         angle2_start, angle2_end,
    #         r_min, r_max,
    #         reverse=False  # 从内到外
    #     )
    #
    #     # ⭐ 拼接形成回路：外1 → 内1 → 内2 → 外2
    #     if seq1.shape[0] > 0 and seq2.shape[0] > 0:
    #         return torch.cat([seq1, seq2], dim=0)
    #     elif seq1.shape[0] > 0:
    #         return seq1
    #     elif seq2.shape[0] > 0:
    #         return seq2
    #     else:
    #         return torch.empty(0, features.shape[0], device=features.device)
    #
    # def _extract_single_sector_sequence(self, features, distances, angles,
    #                                     angle_start, angle_end,
    #                                     r_min, r_max, reverse=False):
    #     """
    #     优化版：减少临时张量创建
    #     """
    #     C, H, W = features.shape
    #     device = features.device
    #
    #     # 1. 创建扇形mask（一次性）
    #     if angle_end > angle_start:
    #         sector_mask = (angles >= angle_start) & (angles < angle_end)
    #     else:
    #         sector_mask = (angles >= angle_start) | (angles < angle_end)
    #
    #     dist_mask = (distances >= r_min) & (distances < r_max)
    #     sector_mask = sector_mask & dist_mask
    #
    #     # ⭐ 立即释放不需要的mask
    #     del dist_mask
    #
    #     if not sector_mask.any():
    #         return torch.empty(0, C, device=device)
    #
    #     # ⭐ 限制半径层数（关键！）
    #     r_min_int = max(0, int(torch.ceil(torch.tensor(r_min)).item()))
    #     r_max_int = int(torch.floor(torch.tensor(r_max)).item())
    #
    #     # ⭐⭐⭐ 硬性限制最多16层
    #     MAX_RADIUS_LAYERS = 16
    #     if r_max_int - r_min_int > MAX_RADIUS_LAYERS:
    #         r_max_int = r_min_int + MAX_RADIUS_LAYERS
    #
    #     if r_min_int >= r_max_int:
    #         return torch.empty(0, C, device=device)
    #
    #     radius_range = range(r_max_int, r_min_int - 1, -1) if reverse else range(r_min_int, r_max_int + 1)
    #
    #     sequence = []
    #
    #     # ⭐ 预先提取扇形区域的特征（避免重复索引）
    #     sector_features = features[:, sector_mask]  # (C, N_pixels_in_sector)
    #     sector_distances = distances[sector_mask]  # (N_pixels_in_sector,)
    #     sector_angles = angles[sector_mask]  # (N_pixels_in_sector,)
    #
    #     # ⭐ 立即释放大的张量
    #     del sector_mask
    #     torch.cuda.empty_cache()
    #
    #     for r in radius_range:
    #         # ⭐ 现在只在小的子集上创建mask
    #         r_mask = (sector_distances >= r - 0.5) & (sector_distances < r + 0.5)
    #
    #         if not r_mask.any():
    #             continue
    #
    #         # 提取当前半径层的像素
    #         pixels_in_layer = sector_features[:, r_mask]  # (C, N_pixels)
    #         angles_in_layer = sector_angles[r_mask]
    #
    #         # 按角度排序
    #         sorted_indices = torch.argsort(angles_in_layer)
    #         pixels_sorted = pixels_in_layer[:, sorted_indices].T  # (N_pixels, C)
    #
    #         sequence.append(pixels_sorted)
    #
    #         # ⭐ 释放临时变量
    #         del r_mask, pixels_in_layer, angles_in_layer, sorted_indices
    #
    #     # ⭐ 清理
    #     del sector_features, sector_distances, sector_angles
    #     torch.cuda.empty_cache()
    #
    #     if len(sequence) > 0:
    #         result = torch.cat(sequence, dim=0)
    #         del sequence
    #         return result
    #     else:
    #         return torch.empty(0, C, device=device)


# ============= 特征重建器 =============

class FeatureReconstructor_MultiScaleSector(nn.Module):
    """重建多尺度扇形特征到2D特征图"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

    def forward(self, sector_results, target_shape):
        """
        Args:
            sector_results: List of dict from MultiScaleSectorFlareModule
            target_shape: (B, C, H, W)
        """
        B, C, H, W = target_shape

        if len(sector_results) == 0:
            return torch.zeros(target_shape, device='cuda' if torch.cuda.is_available() else 'cpu')

        device = sector_results[0]['features'].device
        reconstructed = torch.zeros(B, C, H, W, device=device)
        weight_map = torch.zeros(B, 1, H, W, device=device)

        for result in sector_results:
            source = result['light_source']
            cx, cy = source['center']
            r1 = source['radius_core']
            r2 = source['radius_mid']
            features = result['features']  # (N_sectors, C)
            sector_angle = result['sector_angle']

            # 创建极坐标网格
            y_grid, x_grid = torch.meshgrid(
                torch.arange(H, device=device, dtype=torch.float32),
                torch.arange(W, device=device, dtype=torch.float32),
                indexing='ij'
            )

            dx = x_grid - cx
            dy = y_grid - cy
            distances = torch.sqrt(dx ** 2 + dy ** 2)
            angles = torch.atan2(dy, dx) * 180 / math.pi
            angles = (angles + 360) % 360

            # ⭐ 光斑区域mask
            flare_mask = (distances >= r1) & (distances < r2)

            # 计算扇形索引
            sector_indices = (angles // sector_angle).long()
            num_actual_sectors = features.shape[0]
            sector_indices = sector_indices % num_actual_sectors

            for b in range(B):
                for sector_idx in range(num_actual_sectors):
                    sector_mask = (sector_indices == sector_idx) & flare_mask

                    if sector_mask.any():
                        # ⭐ 直接广播，简洁高效
                        reconstructed[b, :, sector_mask] = features[sector_idx].unsqueeze(-1)
                        weight_map[b, 0, sector_mask] = 1

        # 归一化
        weight_map = torch.clamp(weight_map, min=1)
        reconstructed = reconstructed / weight_map

        return reconstructed

class FlareAttention_MultiScale_Integrated(nn.Module):
    """集成你的光源定位器的FlareAttention"""

    def __init__(self, dim, num_heads=2, reduction_ratio=2,
                 depth_idx=0, base_sector_angle=5,original_size=None, light_sources=None):
        super().__init__()



        # 特征准备
        self.feature_prep = FeaturePreparationWithCBAM(
            in_channels=dim,
            reduction_ratio=reduction_ratio
        )

        prepared_dim = dim // reduction_ratio + 2

        # ⭐ 多尺度扇形模块（自动集成你的检测器）
        self.sector_module = MultiScaleSectorFlareModule_WithDetector(
            in_channels=prepared_dim,
            hidden_dim=dim,
            depth_idx=depth_idx,
            base_sector_angle=base_sector_angle,
            num_heads=num_heads,
            num_layers=2,
            dropout=0.1,
            use_symmetric=True,
            # 光源定位器参数（使用你的配置）
            use_light_detector=False,
            threshold_ratio=0.75,
            base_size=512,
            base_min_area=50,
            base_max_area=None,
            light_sources=None
        )

        # 重建器
        self.reconstructor = FeatureReconstructor_MultiScaleSector(hidden_dim=dim)

        # 融合
        self.fusion = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

    def forward(self, x, x_size, params, light_sources=None):  # ← 添加params参数
        """
        Args:
            x: (B, H*W, C)
            x_size: (H, W)
            params: 参数字典，包含 'original_size' 等
            light_sources: 光源信息
        """
        # 从params获取original_size
        original_size = params.get('original_size', 512)

        B, L, C = x.shape
        H, W = x_size

        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        prepared_features = self.feature_prep(x_2d)

        # 传递light_sources
        results = self.sector_module(prepared_features, light_sources=light_sources)

        if len(results) == 0:
            return x

        flare_feat_map = self.reconstructor(results, (B, C, H, W))
        fused = torch.cat([x_2d, flare_feat_map], dim=1)
        fused = self.fusion(fused)

        return fused.reshape(B, C, L).transpose(1, 2)


#############################################################################fzj_changed#############################################################################

class LPGDownsample(nn.Module):
    """
    空间下采样：以 2×2 为单元，步长 2，仅保留左上角像素。
    通道数不变，高宽减半。
    输入：(B, C, H, W)
    输出：(B, C, H//2, W//2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ✅ 正确：4个维度全部显式写出
        return x[:, :, ::2, ::2]   # B全取, C全取, H隔行, W隔列


def light_sources_Downsample(light_sources, scale_factor=0.5):
    """
    按比例缩放光源信息，兼容 (x, y) 和 (cx, cy, r1) 两种格式。

    Args:
        light_sources: List[List[(x, y)]] 或 List[List[(cx, cy, r1)]]
        scale_factor:  缩放比例（0.5 表示尺寸减半）

    Returns:
        scaled_light_sources: 缩放后的光源信息，格式与输入一致
    """
    if light_sources is None:
        return None

    scaled = []
    for batch_lights in light_sources:
        if batch_lights is None:
            scaled.append(None)
            continue

        scaled_batch = []
        for light_info in batch_lights:
            if len(light_info) == 2:
                # sequences 格式：(x, y)
                x, y = light_info
                scaled_batch.append((
                    x * scale_factor,
                    y * scale_factor,
                ))
            elif len(light_info) == 3:
                # 原始格式：(cx, cy, r1)
                cx, cy, r1 = light_info
                scaled_batch.append((
                    cx * scale_factor,
                    cy * scale_factor,
                    r1 * scale_factor,
                ))
            else:
                raise ValueError(f"light_info 长度应为 2 或 3，实际为 {len(light_info)}")

        scaled.append(scaled_batch)

    return scaled

from scipy.spatial.distance import cdist


class SimpleLightLocator:
    """
    简化版光源定位：基于 LPG 输出检测光源位置和半径（支持批次）

    核心改进：
    1. 中心检测同时考虑峰值点和质心
    2. ⭐ 合并距离相近的小半径光源
    3. ⭐ 自动二次检测应对漏检
    4. ⭐ 支持批次输入 (B, C, H, W) 或 (B, H, W)
    """

    def __init__(self,
                 threshold_ratio=15,
                 threshold_abs=0.3,
                 base_size=512,
                 base_min_area=50,
                 base_max_area=2500,
                 radius_shrink_factor=1,
                 peak_weight=0.5,
                 centroid_weight=0.5,
                 # ⭐ 合并相关参数
                 merge_enabled=True,
                 merge_distance_threshold=50,
                 merge_radius_threshold=80,
                 # ⭐ 二次检测参数
                 enable_second_pass=True,
                 normalization_threshold=0.4,
                 mask_radius_scale=0.9,
                 second_pass_threshold_ratio=None,
                 save_debug_images=False,
                 debug_output_dir=None,
                 verbose=False):

        # 检测参数
        self.threshold_ratio = threshold_ratio
        self.threshold_abs = threshold_abs
        self.base_size = base_size
        self.base_min_area = base_min_area
        self.base_max_area = base_max_area
        self.radius_shrink_factor = radius_shrink_factor

        # 中心检测权重
        assert abs(peak_weight + centroid_weight - 1.0) < 1e-6, \
            "peak_weight + centroid_weight 必须等于 1.0"
        self.peak_weight = peak_weight
        self.centroid_weight = centroid_weight

        # ⭐ 合并参数
        self.merge_enabled = merge_enabled
        self.merge_distance_threshold = merge_distance_threshold
        self.merge_radius_threshold = merge_radius_threshold

        # ⭐ 二次检测参数
        self.enable_second_pass = enable_second_pass
        self.normalization_threshold = normalization_threshold
        self.mask_radius_scale = mask_radius_scale
        self.second_pass_threshold_ratio = second_pass_threshold_ratio
        self.save_debug_images = save_debug_images
        self.debug_output_dir = debug_output_dir

        self.verbose = verbose

    def __call__(self, lpg_map):
        """使对象可以像函数一样调用"""
        return self.detect(lpg_map)

    def detect(self, lpg_map):
        """
        从 LPG 图中检测光源（支持多维度输入）

        Args:
            lpg_map: (B, C, H, W) 或 (B, H, W) 或 (H, W) numpy 数组或 torch.Tensor

        Returns:
            - 2D 输入: List[(cx, cy, r)]
            - 3D/4D 输入: List[List[(cx, cy, r)]]
        """
        # 转换为 numpy
        if torch.is_tensor(lpg_map):
            lpg_np = lpg_map.detach().cpu().numpy()
        else:
            lpg_np = np.array(lpg_map)

        # ⭐ 使用旧版本的简单逻辑（可靠且无歧义）
        if lpg_np.ndim == 4:  # (B, C, H, W)
            B, C, H, W = lpg_np.shape
            if self.verbose:
                print(f"\n🔍 批次输入: {lpg_np.shape} → 返回 List[List[...]]")

            light_infos = []
            for i in range(B):
                # 调整调试输出目录
                original_debug_dir = self.debug_output_dir
                if self.save_debug_images and self.debug_output_dir:
                    self.debug_output_dir = f"{original_debug_dir}/batch_{i:03d}"
                    os.makedirs(self.debug_output_dir, exist_ok=True)

                # 检测单个图像
                info = self._detect_single(lpg_np[i, 0])  # 取第一个通道
                light_infos.append(info)

                # 恢复调试目录
                self.debug_output_dir = original_debug_dir

                if self.verbose:
                    print(f"✅ 批次 {i + 1}/{B} 检测到 {len(info)} 个光源")

            return light_infos  # List[List[...]]

        elif lpg_np.ndim == 3:  # (B, H, W) - ⭐ 无条件当作批次
            B, H, W = lpg_np.shape
            if self.verbose:
                print(f"\n🔍 批次输入: {lpg_np.shape} → 返回 List[List[...]]")

            light_infos = []
            for i in range(B):
                # 调整调试输出目录
                original_debug_dir = self.debug_output_dir
                if self.save_debug_images and self.debug_output_dir:
                    self.debug_output_dir = f"{original_debug_dir}/batch_{i:03d}"
                    os.makedirs(self.debug_output_dir, exist_ok=True)

                # 检测单个图像
                info = self._detect_single(lpg_np[i])
                light_infos.append(info)

                # 恢复调试目录
                self.debug_output_dir = original_debug_dir

                if self.verbose:
                    print(f"✅ 批次 {i + 1}/{B} 检测到 {len(info)} 个光源")

            return light_infos  # List[List[...]]

        elif lpg_np.ndim == 2:  # (H, W) - 单图像
            if self.verbose:
                print(f"\n🔍 单图像输入: {lpg_np.shape} → 返回 List[...]")

            return self._detect_single(lpg_np)  # List[...]

        else:
            raise ValueError(f"不支持的输入维度: {lpg_np.shape}, ndim={lpg_np.ndim}")

    def _detect_single(self, lpg_map):
        """
        单图像检测（内部方法）

        Args:
            lpg_map: (H, W) numpy 数组

        Returns:
            List[(cx, cy, r)]
        """
        # 确保是2D
        if lpg_map.ndim != 2:
            raise ValueError(f"_detect_single 需要2D输入，得到: {lpg_map.shape}")

        # 执行检测
        if self.enable_second_pass:
            final_lights, _, _, _ = self._detect_with_second_pass(lpg_map)
            return final_lights
        else:
            return self._detect_once(lpg_map)

    def _detect_once(self, lpg_map):
        """
        单次检测（底层方法）

        Args:
            lpg_map: (H, W) numpy 数组

        Returns:
            List[(cx, cy, r)]
        """
        H, W = lpg_map.shape

        # 自适应参数
        current_size = min(H, W)
        scale_factor_area = (current_size / self.base_size) ** 2

        min_area = max(int(self.base_min_area * scale_factor_area), 5)
        max_area = int(self.base_max_area * scale_factor_area)

        # 1. 自适应阈值二值化
        mean_intensity = lpg_map.mean()
        std_intensity = lpg_map.std()

        threshold_adaptive = mean_intensity + std_intensity * 2
        threshold_ratio_based = mean_intensity * self.threshold_ratio
        threshold = max(threshold_adaptive, threshold_ratio_based, self.threshold_abs)

        if self.verbose:
            print(f"\n📊 光源检测参数:")
            print(f"   图像尺寸: {W}x{H}")
            print(f"   LPG统计: mean={mean_intensity:.4f}, std={std_intensity:.4f}, max={lpg_map.max():.4f}")
            print(f"   阈值: {threshold:.4f}")
            print(f"   面积范围: [{min_area}, {max_area}]")

        binary = (lpg_map > threshold).astype(np.uint8) * 255

        # 2. 形态学处理
        scale = min(H, W) / self.base_size
        kernel_close_size = max(int(5 * scale), 3)
        kernel_open_size = max(int(3 * scale), 3)

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_close_size, kernel_close_size)
        )
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_open_size, kernel_open_size)
        )

        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)

        # 3. 连通域分析
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if self.verbose:
            print(f"\n🔍 检测到 {len(contours)} 个候选连通域")

        light_infos = []

        for idx, contour in enumerate(contours):
            area = cv2.contourArea(contour)

            # 面积过滤
            if area < min_area or area > max_area:
                if self.verbose:
                    status = "太小" if area < min_area else "太大"
                    print(f"   ❌ 连通域 {idx}: 面积={area:.1f} ({status})")
                continue

            # 4. 同时计算峰值点和质心
            mask = np.zeros((H, W), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # 4.1 峰值点
            masked_lpg = np.where(mask > 0, lpg_map, -np.inf)
            max_idx = np.unravel_index(np.argmax(masked_lpg), masked_lpg.shape)
            cy_peak, cx_peak = max_idx
            max_confidence = lpg_map[cy_peak, cx_peak]

            # 4.2 质心
            M = cv2.moments(contour)
            if M['m00'] == 0:
                continue
            cx_centroid = M['m10'] / M['m00']
            cy_centroid = M['m01'] / M['m00']

            # 4.3 加权组合
            cx = self.peak_weight * cx_peak + self.centroid_weight * cx_centroid
            cy = self.peak_weight * cy_peak + self.centroid_weight * cy_centroid

            # 边界检查
            cx = np.clip(cx, 0, W - 1)
            cy = np.clip(cy, 0, H - 1)

            # 5. 计算半径
            r_equiv = np.sqrt(area / np.pi)
            r = max(r_equiv * self.radius_shrink_factor, 2.0)

            if self.verbose:
                distance = np.sqrt((cx_peak - cx_centroid) ** 2 + (cy_peak - cy_centroid) ** 2)
                print(f"   ✅ 连通域 {idx}:")
                print(f"      峰值点:  ({cx_peak:.1f}, {cy_peak:.1f}), conf={max_confidence:.3f}")
                print(f"      质心:    ({cx_centroid:.1f}, {cy_centroid:.1f})")
                print(f"      最终中心: ({cx:.1f}, {cy:.1f})")
                print(f"      偏差距离: {distance:.1f} pixels")
                print(f"      面积={area:.1f}, r_equiv={r_equiv:.1f}, r={r:.1f}")

            light_infos.append((float(cx), float(cy), float(r)))

        if self.verbose:
            print(f"\n✅ 检测到 {len(light_infos)} 个光源")

        # ⭐ 6. 合并相近的光源
        if self.merge_enabled and len(light_infos) > 1:
            light_infos = self._merge_nearby_lights(light_infos)

        if self.verbose:
            print(f"✅ 最终输出 {len(light_infos)} 个光源\n")

        return light_infos

    def _detect_with_second_pass(self, lpg_map):
        """
        二次检测（内部方法）

        Args:
            lpg_map: (H, W) numpy 数组

        Returns:
            (final_lights, first_pass_lights, second_pass_lights, second_pass_new_lights)
        """
        H, W = lpg_map.shape

        if self.verbose:
            print("\n" + "=" * 60)
            print("🔍 二次检测模式")
            print("=" * 60)

        # 第一次检测
        if self.verbose:
            print("\n【第一次检测】")

        first_pass_lights = self._detect_once(lpg_map)

        if not first_pass_lights:
            if self.verbose:
                print("⚠️ 第一次检测未发现光源")
            return first_pass_lights, first_pass_lights, [], []

        if self.verbose:
            print(f"\n✅ 第一次检测到 {len(first_pass_lights)} 个光源")

        # 保存第一次检测的热力图
        if self.save_debug_images and self.debug_output_dir:
            heatmap_1st = cv2.applyColorMap(
                (lpg_map / lpg_map.max() * 255).astype('uint8'),
                cv2.COLORMAP_JET
            )
            for idx, (cx, cy, r) in enumerate(first_pass_lights):
                cv2.circle(heatmap_1st, (int(cx), int(cy)), int(r), (255, 255, 255), 2)
                cv2.putText(heatmap_1st, f"#{idx + 1}", (int(cx) + 10, int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.imwrite(f"{self.debug_output_dir}/debug_01_first_pass.png", heatmap_1st)

        # 掩盖已检测光源
        if self.verbose:
            print(f"\n【掩盖已检测光源】(半径缩放={self.mask_radius_scale})")

        masked_lpg = self._mask_detected_lights(lpg_map, first_pass_lights)

        non_zero_pixels = (masked_lpg > 0).sum()
        if self.verbose:
            print(f"掩盖后: 非零像素={non_zero_pixels} ({non_zero_pixels / masked_lpg.size * 100:.1f}%)")

        if self.save_debug_images and self.debug_output_dir:
            masked_visual = cv2.applyColorMap(
                (masked_lpg / (masked_lpg.max() + 1e-8) * 255).astype('uint8'),
                cv2.COLORMAP_JET
            )
            cv2.imwrite(f"{self.debug_output_dir}/debug_02_masked.png", masked_visual)

        if non_zero_pixels < 100:
            if self.verbose:
                print("⚠️ 剩余像素太少，跳过第二次检测")
            return first_pass_lights, first_pass_lights, [], []

        # 归一化增强
        if self.verbose:
            print(f"\n【归一化增强】(阈值={self.normalization_threshold})")

        normalized_lpg = self._normalize_for_second_pass(masked_lpg)

        if self.save_debug_images and self.debug_output_dir:
            normalized_visual = cv2.applyColorMap(
                (normalized_lpg * 255).astype('uint8'),
                cv2.COLORMAP_JET
            )
            cv2.imwrite(f"{self.debug_output_dir}/debug_03_normalized.png", normalized_visual)

        # 第二次检测
        if self.verbose:
            print(f"\n【第二次检测】")

        # 临时调整参数
        original_threshold_ratio = self.threshold_ratio
        original_threshold_abs = self.threshold_abs

        if self.second_pass_threshold_ratio is not None:
            self.threshold_ratio = self.second_pass_threshold_ratio
        else:
            self.threshold_ratio = max(original_threshold_ratio * 0.6, 1.8)
        self.threshold_abs = max(original_threshold_abs * 0.5, 0.15)

        second_pass_lights = self._detect_once(normalized_lpg)

        # 恢复参数
        self.threshold_ratio = original_threshold_ratio
        self.threshold_abs = original_threshold_abs

        if not second_pass_lights:
            if self.verbose:
                print("⚠️ 第二次检测未发现新光源")
            return first_pass_lights, first_pass_lights, [], []

        if self.verbose:
            print(f"\n✅ 第二次检测到 {len(second_pass_lights)} 个候选")

        # 过滤重复
        if self.verbose:
            print(f"\n【过滤重复】")

        second_pass_new_lights = self._filter_duplicate_lights(first_pass_lights, second_pass_lights)

        # 合并结果
        final_lights = first_pass_lights + second_pass_new_lights

        if self.verbose:
            print(f"\n" + "=" * 60)
            print(f"📊 检测完成:")
            print(f"   第一次: {len(first_pass_lights)} 个")
            print(f"   新增:   {len(second_pass_new_lights)} 个")
            print(f"   总计:   {len(final_lights)} 个")
            print("=" * 60 + "\n")

        return final_lights, first_pass_lights, second_pass_lights, second_pass_new_lights

    def _merge_nearby_lights(self, light_infos):
        """合并距离相近的小半径光源"""
        if not light_infos:
            return []

        small_lights = []
        large_lights = []

        for light in light_infos:
            cx, cy, r = light
            if r <= self.merge_radius_threshold:
                small_lights.append(light)
            else:
                large_lights.append(light)

        if len(small_lights) <= 1:
            return light_infos

        if self.verbose:
            print(f"\n🔄 合并光源: {len(small_lights)} 个小半径")

        coords = np.array([[light[0], light[1]] for light in small_lights])
        n = len(small_lights)

        parent = list(range(n))

        def find(x):
            if parent[x] != x:
                parent[x] = find(parent[x])
            return parent[x]

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                parent[px] = py

        distances = cdist(coords, coords)

        for i in range(n):
            for j in range(i + 1, n):
                if distances[i, j] <= self.merge_distance_threshold:
                    if find(i) != find(j):
                        union(i, j)

        groups = {}
        for i in range(n):
            root = find(i)
            if root not in groups:
                groups[root] = []
            groups[root].append(i)

        merged_lights = []

        for group_indices in groups.values():
            if len(group_indices) == 1:
                merged_lights.append(small_lights[group_indices[0]])
            else:
                group_lights = [small_lights[i] for i in group_indices]
                cx_merged = np.mean([light[0] for light in group_lights])
                cy_merged = np.mean([light[1] for light in group_lights])

                max_radius = 0
                for light in group_lights:
                    cx, cy, r = light
                    dist = np.sqrt((cx - cx_merged) ** 2 + (cy - cy_merged) ** 2)
                    max_radius = max(max_radius, dist + r)

                merged_lights.append((
                    float(cx_merged),
                    float(cy_merged),
                    float(max_radius),
                ))

        merged_lights.extend(large_lights)

        if self.verbose:
            print(f"   合并后: {len(merged_lights)} 个")

        return merged_lights

    def _mask_detected_lights(self, lpg_map, light_infos):
        """掩盖已检测光源"""
        H, W = lpg_map.shape
        masked_lpg = lpg_map.copy()
        mask = np.zeros((H, W), dtype=np.uint8)

        for cx, cy, r in light_infos:
            mask_radius = int(r * self.mask_radius_scale)
            cv2.circle(mask, (int(cx), int(cy)), mask_radius, 255, -1)

        masked_lpg[mask > 0] = 0
        return masked_lpg

    def _normalize_for_second_pass(self, masked_lpg):
        """归一化增强"""
        mask_below = masked_lpg <= self.normalization_threshold
        values_below = masked_lpg[mask_below]

        if len(values_below) == 0:
            min_val = masked_lpg.min()
            max_val = masked_lpg.max()
        else:
            min_val = values_below.min()
            max_val = values_below.max()

        if max_val - min_val < 1e-8:
            return masked_lpg

        normalized = (masked_lpg - min_val) / (max_val - min_val)
        return np.clip(normalized, 0, 1)

    def _filter_duplicate_lights(self, first_pass_lights, second_pass_lights):
        """过滤重复光源"""
        if not first_pass_lights:
            return second_pass_lights

        first_coords = np.array([[light[0], light[1]] for light in first_pass_lights])
        new_lights = []

        for light in second_pass_lights:
            cx, cy, r = light
            distances = np.sqrt(
                (first_coords[:, 0] - cx) ** 2 +
                (first_coords[:, 1] - cy) ** 2
            )
            if distances.min() > self.merge_distance_threshold:
                new_lights.append(light)

        if self.verbose:
            print(f"   过滤: {len(second_pass_lights)} → {len(new_lights)} 个新光源")

        return new_lights


import matplotlib
matplotlib.use('Agg')  # 强制使用无 GUI 的纯图像渲染后端，必须在导入 pyplot 之前调用！
import matplotlib.pyplot as plt
@ARCH_REGISTRY.register()
class DeflareMambav2(nn.Module):
    r""" MambaIRv2 Model
           A PyTorch impl of : `A Simple Baseline for Image Restoration with State Space Model `.

       Args:
           img_size (int | tuple(int)): Input image size. Default 64
           patch_size (int | tuple(int)): Patch size. Default: 1
           in_chans (int): Number of input image channels. Default: 3
           embed_dim (int): Patch embedding dimension. Default: 96
           d_state (int): num of hidden state in the state space model. Default: 16
           depths (tuple(int)): Depth of each RSSG
           drop_rate (float): Dropout rate. Default: 0
           drop_path_rate (float): Stochastic depth rate. Default: 0.1
           norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
           patch_norm (bool): If True, add normalization after patch embedding. Default: True
           use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
           upscale: Upscale factor. 2/3/4 for image SR, 1 for denoising
           img_range: Image range. 1. or 255.
           upsampler: The reconstruction reconstruction module. 'pixelshuffle'/None
           resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
       """

    def __init__(self,
                 img_size=512,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=36,
                 depths=(2, 2, 2, 2, 2, 2, 2),
                 drop_rate=0.,
                 d_state=16,
                 mlp_ratio=2.,  ### expand
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 num_heads=(2, 2, 2, 2, 2, 2, 2),
                 window_size=16,
                 inner_rank=32,
                 num_tokens=64,
                 convffn_kernel_size=5,
                 qkv_bias=True,
                 **kwargs):
        super(DeflareMambav2, self).__init__()
        num_in_ch = 3  # 3+1
        num_out_ch = 6
        num_feat = 64
        self.img_range = img_range

        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)

        self.num_heads = num_heads
        self.window_size = window_size
        self.inner_rank = inner_rank
        self.num_tokens = num_tokens
        self.convffn_kernel_size = convffn_kernel_size
        self.qkv_bias = qkv_bias
        self.upscale = upscale
        self.upsampler = upsampler
        self.mlp_ratio = mlp_ratio
        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        self.is_light_sr = True if self.upsampler=='pixelshuffledirect' else False
        # ------------------------- 2, deep feature extraction ------------------------- #
        # self.num_layers = len(depths)
        self.num_enc_layers = 3
        self.num_dec_layers = 3
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm

        # # stochastic depth
        # enc_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths[:self.num_enc_layers]))]
        # conv_dpr = [drop_path_rate] * depths[3]
        # dec_dpr = enc_dpr[::-1]
        # # refine_dpr = [drop_path_rate] * depths[7]

        # 应该修改为：
        # 计算所有层的总深度
        total_depth = sum(depths)
        # 为所有层生成统一的drop_path率
        total_dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_depth)]
        # 然后按层分配
        self.enc0_dpr = total_dpr[sum(depths[:0]):sum(depths[:1])]
        self.enc1_dpr = total_dpr[sum(depths[:1]):sum(depths[:2])]
        self.enc2_dpr = total_dpr[sum(depths[:2]):sum(depths[:3])]
        self.bottle_dpr = [drop_path_rate] * depths[3]  # 瓶颈层使用固定的drop_path_rate
        self.dec1_dpr = total_dpr[sum(depths[:4]):sum(depths[:5])]
        self.dec2_dpr = total_dpr[sum(depths[:5]):sum(depths[:6])]
        self.dec3_dpr = total_dpr[sum(depths[:6]):sum(depths[:7])]


        self.LPGDownsample = LPGDownsample()
        self.light_sources_Downsample = light_sources_Downsample


        self.enc0_patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.enc0_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.enc0_patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)


        # 编码器层0
        self.enc0_layer = ASSBWrapper(
            dim=embed_dim,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[0],
            drop_path=self.enc0_dpr,
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            num_heads=num_heads[0],
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            qkv_bias=qkv_bias,
            idx=0,  # 层索引
            light_sources=None
        )

        # 类似地替换其他层：enc1_layer, enc2_layer, bottle_layer, dec1_layer等
        # 记得为每一层提供正确的idx
        self.enc0_norm = norm_layer(embed_dim)
        self.downsample_0 = nn.Conv2d(embed_dim, embed_dim*2, kernel_size=4, stride=2, padding=1)

        self.enc1_patch_embed = PatchEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim * 2,
            embed_dim=embed_dim * 2,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.enc1_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.enc1_patch_unembed = PatchUnEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim * 2,
            embed_dim=embed_dim * 2,
            norm_layer=norm_layer if self.patch_norm else None)


        # 编码器层1
        self.enc1_layer = ASSBWrapper(
            dim=embed_dim * 2,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[1],
            drop_path=self.enc1_dpr,
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size//2,
            patch_size=patch_size,
            resi_connection=resi_connection,
            num_heads=num_heads[1],
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            qkv_bias=qkv_bias,
            idx=1,  # 层索引
            light_sources=None
        )


        self.enc1_norm = norm_layer(embed_dim * 2)
        self.downsample_1 = nn.Conv2d(embed_dim*2, embed_dim*4, kernel_size=4, stride=2, padding=1)

        self.enc2_patch_embed = PatchEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim * 4,
            embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.enc2_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.enc2_patch_unembed = PatchUnEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim * 4,
            embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None)


        # 编码器层2
        self.enc2_layer = ASSBWrapper(
            dim=embed_dim * 4,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[2],
            drop_path=self.enc2_dpr,
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size // 4,
            patch_size=patch_size,
            resi_connection=resi_connection,
            num_heads=num_heads[2],
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            qkv_bias=qkv_bias,
            idx=2,  # 层索引
            light_sources=None
        )

        self.enc2_norm = norm_layer(embed_dim * 4)
        self.downsample_2 = nn.Conv2d(embed_dim*4, embed_dim*8, kernel_size=4, stride=2, padding=1)


        self.bottle_patch_embed = PatchEmbed(
            img_size=img_size // 8,
            patch_size=patch_size,
            in_chans=embed_dim * 8,
            embed_dim=embed_dim * 8,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.bottle_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.bottle_patch_unembed = PatchUnEmbed(
            img_size=img_size // 8,
            patch_size=patch_size,
            in_chans=embed_dim * 8,
            embed_dim=embed_dim * 8,
            norm_layer=norm_layer if self.patch_norm else None)



        self.bottle_layer = ASSBWrapper(
            dim=embed_dim * 8,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[3],
            drop_path=self.bottle_dpr,
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size // 8,
            patch_size=patch_size,
            resi_connection=resi_connection,
            num_heads=num_heads[3],
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            qkv_bias=qkv_bias,
            idx=3,  # 层索引
            light_sources=None
        )



        self.bottle_norm = norm_layer(embed_dim * 8)



        self.upsample_1 = nn.ConvTranspose2d(embed_dim * 8, embed_dim * 4, kernel_size=2, stride=2)
        self.dec1_patch_embed = PatchEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim * 8,
            embed_dim=embed_dim * 8,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.dec1_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.dec1_patch_unembed = PatchUnEmbed(
            img_size=img_size // 4,
            patch_size=patch_size,
            in_chans=embed_dim * 8,
            embed_dim=embed_dim * 8,
            norm_layer=norm_layer if self.patch_norm else None)


        self.dec1_layer = ASSBWrapper(
            dim=embed_dim * 8,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[4],
            drop_path=self.dec1_dpr,
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size // 4,
            patch_size=patch_size,
            resi_connection=resi_connection,
            num_heads=num_heads[4],
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            qkv_bias=qkv_bias,
            idx=4,  # 层索引
            light_sources=None
        )
        self.dec1_norm = norm_layer(embed_dim * 8)

        self.upsample_2 = nn.ConvTranspose2d(embed_dim * 8, embed_dim * 2, kernel_size=2, stride=2)
        self.dec2_patch_embed = PatchEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim * 4,
            embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.dec2_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.dec2_patch_unembed = PatchUnEmbed(
            img_size=img_size // 2,
            patch_size=patch_size,
            in_chans=embed_dim * 4,
            embed_dim=embed_dim * 4,
            norm_layer=norm_layer if self.patch_norm else None)



        self.dec2_layer = ASSBWrapper(
            dim=embed_dim * 4,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[5],
            drop_path=self.dec2_dpr,
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size // 2,
            patch_size=patch_size,
            resi_connection=resi_connection,
            num_heads=num_heads[5],
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            qkv_bias=qkv_bias,
            idx=5,  # 层索引
            light_sources=None
        )
        self.dec2_norm = norm_layer(embed_dim * 4)

        self.upsample_3 = nn.ConvTranspose2d(embed_dim * 4, embed_dim, kernel_size=2, stride=2)
        self.dec3_patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim*2,
            embed_dim=embed_dim*2,
            norm_layer=norm_layer if self.patch_norm else None)
        # num_patches = self.patch_embed.num_patches
        patches_resolution = self.dec3_patch_embed.patches_resolution
        # self.patches_resolution[i_enc_layer] = patches_resolution
        # return 2D feature map from 1D token sequence
        self.dec3_patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim*2,
            embed_dim=embed_dim*2,
            norm_layer=norm_layer if self.patch_norm else None)


        self.dec3_layer = ASSBWrapper(
            dim=embed_dim * 2,
            input_resolution=(patches_resolution[0], patches_resolution[1]),
            depth=depths[6],
            drop_path=self.dec3_dpr,
            d_state=d_state,
            mlp_ratio=self.mlp_ratio,
            norm_layer=norm_layer,
            downsample=None,
            use_checkpoint=use_checkpoint,
            img_size=img_size,
            patch_size=patch_size,
            resi_connection=resi_connection,
            num_heads=num_heads[6],
            window_size=window_size,
            inner_rank=inner_rank,
            num_tokens=num_tokens,
            convffn_kernel_size=convffn_kernel_size,
            qkv_bias=qkv_bias,
            idx=6,  # 层索引
            light_sources=None
        )

        self.dec3_norm = norm_layer(embed_dim*2)




        self.pos_drop = nn.Dropout(p=drop_rate)
        self.is_light_sr = True if self.upsampler == 'pixelshuffledirect' else False
        # stochastic depth

        # -------------------------3. high-quality image reconstruction ------------------------ #
        self.conv_last = nn.Conv2d(embed_dim*2, num_out_ch, 3, 1, 1)
        # self.conv_deflare = nn.Conv2d(3, 3, 3, 1, 1)
        # self.conv_flare = nn.Conv2d(3, 3, 3, 1, 1)


        self.apply(self._init_weights)
        self.activation = nn.Sigmoid()

        # -------------------------light-locator ------------------------ #
        self.locator = SimpleLightLocator(
            threshold_ratio=15.0,
            threshold_abs=0.3,
            base_size=512,
            base_min_area=50,
            base_max_area=2500,
            radius_shrink_factor=1,
            peak_weight=0.5,
            centroid_weight=0.5,
            verbose=False  # 开启详细模式
        )




    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, patch_embed, patch_unembed, norm, layer,
                         light_sources=None, lpg=None, lpg_first_stage=None):
        """
        前向传播特征提取

        Args:
            x: 输入特征 (B, C, H, W)
            patch_embed: patch嵌入层
            patch_unembed: patch反嵌入层
            norm: 归一化层
            layer: ASSBWrapper层
            light_sources: 光源信息 List[List[(cx, cy, r1, r2)]]，默认为None
            lpg: 最终LPG (B, 1, H, W)
            lpg_first_stage: 第1次LPG (B, 1, H, W)

        Returns:
            x: 输出特征 (B, C, H, W)
        """
        x_size = (x.shape[2], x.shape[3])
        x = patch_embed(x)  # (B, L, C)

        # ⭐ 传递 light_sources、lpg、lpg_first_stage
        if hasattr(layer, 'assb'):
            x = layer(x, x_size, light_sources=light_sources,
                      lpg=lpg, lpg_first_stage=lpg_first_stage)
        else:
            x = layer(x, x_size)

        x = norm(x)
        x = patch_unembed(x, x_size)
        return x

    def pos_map_to_sequences(self,
                             pos_map: torch.Tensor,
                             conf_thresh: float = 0.05,
                             nms_kernel: int = 9,
                             ) -> list[list[tuple[float, float]]]:
        """
        将 pos_map 图像还原成位置链表。

        训练时 max_lights=1（只取最强光源，减少噪声干扰）
        测试时 max_lights=6（提取所有有效光源）
        """
        # ── 根据训练/测试模式自动设置 max_lights ─────────────────────────
        max_lights = 1 if self.training else 6

        B, _, H, W = pos_map.shape
        device = pos_map.device
        pad = nms_kernel // 2

        # ── Step1：MaxPool NMS ─────────────────────────────────────────────
        hm_max = F.max_pool2d(pos_map, nms_kernel, stride=1, padding=pad)
        peaks = (pos_map == hm_max) & (pos_map > conf_thresh)

        # ── Step2：归一化坐标网格 ──────────────────────────────────────────
        xs_grid = torch.linspace(0, 1, W, device=device)
        ys_grid = torch.linspace(0, 1, H, device=device)

        sequences = []

        for b in range(B):
            peak_mask = peaks[b, 0]
            scores = pos_map[b, 0] * peak_mask

            n_peaks = int(peak_mask.sum().item())
            if n_peaks == 0:
                sequences.append([])
                continue

            topk_vals, topk_idx = scores.view(-1).topk(
                min(max_lights, n_peaks), largest=True, sorted=True
            )

            sample_lights = []
            for val, idx in zip(topk_vals, topk_idx):
                if float(val) < conf_thresh:
                    break
                y_idx = int(idx) // W
                x_idx = int(idx) % W
                x = round(float(xs_grid[x_idx]), 4)
                y = round(float(ys_grid[y_idx]), 4)
                sample_lights.append((x, y))

            sequences.append(sample_lights)

        return sequences

    def get_flare_guidance(self,
                           heatmap_full: torch.Tensor,
                           flare_map: torch.Tensor,
                           threshold: float = 0.2
                           ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        从 heatmap_full 和 flare_map 生成 light_mask 和 flare_guidance。

        Args:
            heatmap_full: [B, 1, H, W]  光源热力图，值域 [0, 1]
            flare_map:    [B, 1, H, W]  炫光掩码图，值域 [0, 1]
            threshold:    二值化阈值，默认 0.3

        Returns:
            light_mask:     [B, 1, H, W]  二值化光源掩码，0 或 1
            flare_guidance: [B, 1, H, W]  flare_map 去除光源区域后归一化结果，值域 [0, 1]
        """
        # ── Step1：heatmap 二值化 → light_mask（保持不变）─────────────────
        light_mask = (heatmap_full >= threshold).float()  # [B, 1, H, W]

        # ── Step2：将 light_mask 为 1 的位置在 flare_map 上置 0 ───────────
        # flare_guidance = flare_map * (1.0 - light_mask)  # [B, 1, H, W]
        flare_guidance = flare_map   # [B, 1, H, W]
        # ── Step3：逐样本 min-max 归一化到 [0, 1] ─────────────────────────
        B = flare_guidance.shape[0]
        fg_flat = flare_guidance.view(B, -1)  # [B, H*W]
        fg_min = fg_flat.min(dim=1).values.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        fg_max = fg_flat.max(dim=1).values.view(B, 1, 1, 1)  # [B, 1, 1, 1]
        # 防止全图为 0 时除以 0
        flare_guidance = (flare_guidance - fg_min) / (fg_max - fg_min + 1e-6)

        return light_mask, flare_guidance

    def _save_visualization(self, x, pos_map, heatmap_full, flare_map,
                            light_mask, flare_guidance, light_sequences,
                            iter_idx, save_path):
        """
        保存中间特征图的可视化结果
        """
        # 确保保存路径存在
        os.makedirs(save_path, exist_ok=True)

        # 为了可视化，我们只取 Batch 中的第一张图 (index 0)
        b_idx = 0

        # 1. 转换张量为 numpy 格式
        # 原图 [3, H, W] -> [H, W, 3]，并限制在 0-1 之间防止色彩溢出
        img_np = x[b_idx].detach().cpu().permute(1, 2, 0).numpy()
        img_np = np.clip(img_np, 0, 1)

        # 单通道特征图 [1, H, W] -> [H, W]
        pos_np = pos_map[b_idx, 0].detach().cpu().numpy()
        heat_np = heatmap_full[b_idx, 0].detach().cpu().numpy()
        flare_np = flare_map[b_idx, 0].detach().cpu().numpy()
        mask_np = light_mask[b_idx, 0].detach().cpu().numpy()
        guid_np = flare_guidance[b_idx, 0].detach().cpu().numpy()

        # 提取当前图片的光源坐标
        points = light_sequences[b_idx]
        H, W = img_np.shape[:2]

        # 2. 创建 2x3 的子图画布
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        plt.suptitle(f"Iteration: {iter_idx}", fontsize=16)

        # (1, 1) 原图 + 光源位置标记
        axes[0, 0].imshow(img_np)
        for (px, py) in points:
            # 你的代码中 x, y 是 0-1 归一化的，这里还原到像素坐标
            axes[0, 0].scatter(px * W, py * H, c='red', marker='+', s=100, linewidths=2)
        axes[0, 0].set_title("Original Image & Light Points")
        axes[0, 0].axis('off')

        # (1, 2) 光源位置高斯图
        im1 = axes[0, 1].imshow(pos_np, cmap='viridis')
        axes[0, 1].set_title("pos_map (Gaussian)")
        axes[0, 1].axis('off')
        fig.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

        # (1, 3) 光源热力图
        im2 = axes[0, 2].imshow(heat_np, cmap='hot')
        axes[0, 2].set_title("heatmap_full")
        axes[0, 2].axis('off')
        fig.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

        # (2, 1) 炫光掩码图
        im3 = axes[1, 0].imshow(flare_np, cmap='gray')
        axes[1, 0].set_title("flare_map")
        axes[1, 0].axis('off')
        fig.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04)

        # (2, 2) 光源掩码图 (二值化)
        im4 = axes[1, 1].imshow(mask_np, cmap='gray')
        axes[1, 1].set_title("light_mask (Binary)")
        axes[1, 1].axis('off')
        fig.colorbar(im4, ax=axes[1, 1], fraction=0.046, pad=0.04)

        # (2, 3) 炫光引导图
        im5 = axes[1, 2].imshow(guid_np, cmap='plasma')
        axes[1, 2].set_title("flare_guidance")
        axes[1, 2].axis('off')
        fig.colorbar(im5, ax=axes[1, 2], fraction=0.046, pad=0.04)

        # 3. 保存图片并关闭画布(释放内存)
        plt.tight_layout()
        file_name = os.path.join(save_path, f"vis_features_iter_{iter_idx:06d}.png")
        plt.savefig(file_name, dpi=150, bbox_inches='tight')
        plt.close(fig)  # 非常重要！防止训练时内存泄漏


    def forward(self, x, iter_idx=None, enable_vis=False, vis_save_path=None):
        # ── 分离 IPN 输出和原图 ────────────────────────────────────────────
        pos_map = x[:, -1:, :, :]  # [B, 1, H, W]  光源位置高斯图
        heatmap_full = x[:, -2:-1, :, :]  # [B, 1, H, W]  光源热力图
        flare_map = x[:, -3:-2, :, :]  # [B, 1, H, W]  炫光掩码图
        x = x[:, :-3, :, :]  # [B, 3, H, W]  原始图像

        # ── pos_map → 位置链表 ─────────────────────────────────────────────
        light_sequences = self.pos_map_to_sequences(
            pos_map,
            conf_thresh=0.4,  # 高斯图峰值阈值，按实际效果调整
            nms_kernel=9,  # NMS 窗口
        )
        light_mask, flare_guidance = self.get_flare_guidance(heatmap_full, flare_map)
 


        self.mean = self.mean.type_as(x)
        x_mean = (x - self.mean) * self.img_range
        y = self.conv_first(x_mean)
        y = self.pos_drop(y)

        # === Encoder Level 0 ===
        conv0 = self.forward_features(y, self.enc0_patch_embed, self.enc0_patch_unembed,
                                      self.enc0_norm, self.enc0_layer,light_sources=light_sequences,lpg=light_mask,lpg_first_stage=flare_guidance)
        pool0 = self.downsample_0(conv0)

        # === Level 1 ===
        lpg1 = self.LPGDownsample(light_mask)
        lpg_first_stage1 = self.LPGDownsample(flare_guidance)


        light_infos1 = self.light_sources_Downsample(light_sequences, scale_factor=0.5)  # 1/2 分辨率


        # ⭐ 立即可视化 Level 1
        # if enable_vis and iter_idx is not None:
        #     self._visualize_level(1, iter_idx, vis_save_path)

        # === Encoder Level 1 ===
        conv1 = self.forward_features(pool0, self.enc1_patch_embed, self.enc1_patch_unembed,
                                      self.enc1_norm, self.enc1_layer, light_sources=light_infos1,lpg=lpg1,lpg_first_stage=lpg_first_stage1)
        pool1 = self.downsample_1(conv1)

        # === Level 2 ===
        lpg2 = self.LPGDownsample(lpg1)
        lpg_first_stage2 = self.LPGDownsample(lpg_first_stage1)


        light_infos2 = self.light_sources_Downsample(light_sequences, scale_factor=0.25)  # 1/4 分辨率

        # ⭐ 立即可视化 Level 2
        # if enable_vis and iter_idx is not None:
        #     self._visualize_level(2, iter_idx, vis_save_path)

        # === Encoder Level 2 ===
        conv2 = self.forward_features(pool1, self.enc2_patch_embed, self.enc2_patch_unembed,
                                      self.enc2_norm, self.enc2_layer, light_sources=light_infos2,lpg=lpg2, lpg_first_stage=lpg_first_stage2)
        pool2 = self.downsample_2(conv2)

        # === Level 3 ===
        lpg3 = self.LPGDownsample(lpg2)
        lpg_first_stage3 = self.LPGDownsample(lpg_first_stage2)


        light_infos3 = self.light_sources_Downsample(light_sequences, scale_factor=0.125)  # 1/8 分辨率

        # ⭐ 立即可视化 Level 3
        # if enable_vis and iter_idx is not None:
        #     self._visualize_level(3, iter_idx, vis_save_path)

        # === Bottleneck ===
        conv3 = self.forward_features(pool2, self.bottle_patch_embed, self.bottle_patch_unembed,
                                      self.bottle_norm, self.bottle_layer, light_sources=light_infos3,lpg=lpg3, lpg_first_stage=lpg_first_stage3)

        # === Decoder ===
        up1 = self.upsample_1(conv3)
        deconv1 = torch.cat((up1, conv2), dim=1)
        deconv1 = self.forward_features(deconv1, self.dec1_patch_embed, self.dec1_patch_unembed,
                                        self.dec1_norm, self.dec1_layer, light_sources=light_infos2,lpg=lpg2, lpg_first_stage=lpg_first_stage2)

        up2 = self.upsample_2(deconv1)
        deconv2 = torch.cat((up2, conv1), dim=1)
        deconv2 = self.forward_features(deconv2, self.dec2_patch_embed, self.dec2_patch_unembed,
                                        self.dec2_norm, self.dec2_layer, light_sources=light_infos1,lpg=lpg1, lpg_first_stage=lpg_first_stage1)

        up3 = self.upsample_3(deconv2)
        deconv3 = torch.cat((up3, conv0), dim=1)
        deconv3 = self.forward_features(deconv3, self.dec3_patch_embed, self.dec3_patch_unembed,
                                        self.dec3_norm, self.dec3_layer, light_sources=light_sequences,lpg=light_mask, lpg_first_stage=flare_guidance)

        y = self.conv_last(deconv3)


        y = self.activation(y)


        return y







class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch):
        self.num_feat = num_feat
        m = []
        m.append(nn.Conv2d(num_feat, (scale**2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)



class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


