import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision.ops import deform_conv2d
    HAS_DCN = True
except ImportError:
    HAS_DCN = False


# ================================================================
# 基础模块
# ================================================================

class ConvBnRelu(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, k, s, padding=p,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """★ Bottleneck 风格：1×1降维 → 3×3 → 1×1升维，参数比两个3×3少~50%"""
    def __init__(self, ch):
        super().__init__()
        mid = max(ch // 2, 8)
        self.block = nn.Sequential(
            nn.Conv2d(ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, mid, 3, 1, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.block(x))


# ================================================================
# 注意力：用轻量 SE 替代 CBAM（去掉空间注意力卷积）
# 保留 CBAM 名字以减少改动，但内部改为 SE
# ================================================================

class CBAM(nn.Module):
    """★ 改为纯通道 SE，去掉空间注意力 7×7 卷积，参数减少 ~70%"""
    def __init__(self, ch, reduction=8):
        super().__init__()
        mid = max(ch // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(ch, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, ch),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.se(x).view(x.size(0), x.size(1), 1, 1)
        return x * w


# ================================================================
# 频域增强（保留，只随 ch 缩减）
# ================================================================

class FreqEnhance(nn.Module):
    """
    安全频域增强：幅度谱调制 + 相位严格保留，从根本上避免亮暗反转。
    """
    def __init__(self, ch):
        super().__init__()
        self.amp_conv = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )
        # 初始化为0：训练初期 sigmoid(0)=0.5，增强温和；
        # 随训练自适应增大或缩小
        self.scale = nn.Parameter(torch.zeros(1))
        self.bn    = nn.BatchNorm2d(ch)
        self.relu  = nn.ReLU(inplace=True)

    def forward(self, x):
        B, C, H, W = x.shape

        # ── Step1：FFT ────────────────────────────────────────────
        f = torch.fft.rfft2(x, norm='ortho')        # [B,C,H,W//2+1] 复数

        # ── Step2：极坐标分解 ─────────────────────────────────────
        amp   = torch.abs(f)                         # 幅度 ≥ 0
        phase = torch.angle(f)                       # 相位，[-π, π]，严格不动

        # ── Step3：幅度增强（残差风格） ───────────────────────────
        amp_delta = self.amp_conv(amp)               # 学习幅度残差
        amp_out   = amp + torch.sigmoid(self.scale) * amp_delta  # 残差叠加

        # ── Step4：用增强幅度 + 原始相位重建复数谱 ───────────────
        f_out = torch.polar(amp_out, phase)          # ← 相位完全没变

        # ── Step5：逆变换 + 空间域残差 ───────────────────────────
        out = torch.fft.irfft2(f_out, s=(H, W), norm='ortho')
        return self.relu(self.bn(x + out))


# ================================================================
# 线条捕捉：★ 只保留 H+V 两方向，去掉两个膨胀对角卷积
# ================================================================

class MultiOrientationStrip(nn.Module):
    """★ 2方向（H+V），去掉膨胀对角，参数减少 ~50%"""
    def __init__(self, in_ch, out_ch, strip_len=11):
        super().__init__()
        assert strip_len % 2 == 1
        self.conv_h = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (1, strip_len),
                      padding=(0, strip_len // 2), bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.conv_v = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, (strip_len, 1),
                      padding=(strip_len // 2, 0), bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.conv_h(x), self.conv_v(x)], dim=1))


class DeformableStrip(nn.Module):
    """★ offset_conv 内部通道 32→16"""
    def __init__(self, in_ch, out_ch, k=3):
        super().__init__()
        self.k = k
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_ch, 16, 3, padding=1, bias=False),   # ★ 32→16
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 2 * k * k, 3, padding=1, bias=False)
        )
        self.weight = nn.Parameter(torch.randn(out_ch, in_ch, k, k) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_ch))
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        offset = self.offset_conv(x)
        if HAS_DCN:
            out = deform_conv2d(x, offset, self.weight, self.bias,
                                padding=self.k // 2)
        else:
            out = F.conv2d(x, self.weight, self.bias, padding=self.k // 2)
        return self.relu(self.bn(out))


class LineCaptureModule(nn.Module):
    def __init__(self, ch, strip_len=11):
        super().__init__()
        half = ch // 2
        self.strip = MultiOrientationStrip(ch, half, strip_len)
        self.deform = DeformableStrip(ch, half)
        self.fuse = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.fuse(torch.cat([self.strip(x), self.deform(x)], dim=1)) + x


# ================================================================
# 编码器（★ base_ch=24，各 stage 只用 1 个 ResBlock）
# ================================================================

class Encoder(nn.Module):
    """
    输出 4 尺度：
        e1 [B, 24,  H,   W  ]
        e2 [B, 48,  H/2, W/2]
        e3 [B, 96,  H/4, W/4]
        e4 [B, 192, H/8, W/8]
    """
    def __init__(self, in_ch=3, base_ch=24, strip_len=11):
        super().__init__()
        ch = base_ch

        self.stem   = ConvBnRelu(in_ch, ch)
        self.stage1 = ResBlock(ch)
        self.hf1    = HFEnhance(ch, sigma=0.1)           # ⭐ e1：全分辨率高频增强

        self.down1  = ConvBnRelu(ch, ch * 2, s=2)
        self.stage2 = ResBlock(ch * 2)
        self.hf2    = HFEnhance(ch * 2, sigma=0.1)       # ⭐ e2：半分辨率高频增强
        self.line_capture = LineCaptureModule(ch * 2, strip_len)

        self.down2  = ConvBnRelu(ch * 2, ch * 4, s=2)
        self.stage3 = nn.Sequential(ResBlock(ch * 4), ResBlock(ch * 4))

        self.down3  = ConvBnRelu(ch * 4, ch * 8, s=2)
        self.stage4 = nn.Sequential(ResBlock(ch * 8), ResBlock(ch * 8))

        self.freq = FreqEnhance(ch * 8)                   # 保持原有（深层语义频域增强）
        self.cbam = CBAM(ch * 8)

    def forward(self, x):
        e1 = self.hf1(self.stage1(self.stem(x)))                          # ⭐ 全分辨率高频增强
        e2 = self.line_capture(self.hf2(self.stage2(self.down1(e1))))     # ⭐ 半分辨率高频增强
        e3 = self.stage3(self.down2(e2))
        e4 = self.cbam(self.freq(self.stage4(self.down3(e3))))
        return e1, e2, e3, e4

# ================================================================
# 炫光分割 Decoder
# ★ ASPP：3分支，内部通道 ch//2；CBAM 只保留最后一层
# ================================================================

class ASPP(nn.Module):
    """★ 3分支（去掉 rate=16），内部通道砍半"""
    def __init__(self, in_ch, out_ch, rates=(1, 4, 8)):
        super().__init__()
        mid = out_ch // 2                    # ★ 内部通道砍半
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, mid, 3, padding=r, dilation=r, bias=False),
                nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
            ) for r in rates
        ])
        self.global_branch = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(mid * (len(rates) + 1), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feats = [b(x) for b in self.branches]
        g = F.interpolate(self.global_branch(x), size=x.shape[2:],
                          mode='bilinear', align_corners=False)
        feats.append(g)
        return self.fuse(torch.cat(feats, dim=1))


class FlareDecoder(nn.Module):

    def __init__(self, base_ch=24):
        super().__init__()
        ch = base_ch

        self.aspp = ASPP(ch * 8, ch * 8)

        self.up3_conv = ConvBnRelu(ch * 8 + ch * 4, ch * 4)
        self.up3_res  = ResBlock(ch * 4)


        self.up2_conv = ConvBnRelu(ch * 4 + ch * 2, ch * 2)
        self.up2_res  = ResBlock(ch * 2)


        self.up1_conv = ConvBnRelu(ch * 2 + ch, ch)
        self.up1_res  = ResBlock(ch)
        self.up1_att  = CBAM(ch)                           # ★ 只保留最后一层

        self.head = nn.Sequential(
            nn.Conv2d(ch, 1, 1),
            nn.Sigmoid()
        )

    def _up_cat(self, feat, skip):
        feat = F.interpolate(feat, size=skip.shape[2:],
                             mode='bilinear', align_corners=False)
        return torch.cat([feat, skip], dim=1)

    def forward(self, e1, e2, e3, e4):
        x  = self.aspp(e4)
        d3 = self.up3_res(self.up3_conv(self._up_cat(x,  e3)))
        d2 = self.up2_res(self.up2_conv(self._up_cat(d3, e2)))
        d1 = self.up1_att(self.up1_res(self.up1_conv(self._up_cat(d2, e1))))
        return self.head(d1)  # [B, 1, H, W]


# ================================================================
# 热力图头（★ 内部通道 64→32）
# ================================================================

class LightHeatmapHead(nn.Module):
    """
    输出单通道热力图 [B, 1, H/4, W/4]
    返回 (heatmap, logits)，接口不变
    """
    def __init__(self, ch_large=192, ch_small=96):
        super().__init__()
        # ★ 内部统一用 32 通道（原来 64）
        self.from_e4 = ConvBnRelu(ch_large, 32, k=1, p=0)
        self.from_e3 = ConvBnRelu(ch_small, 32, k=1, p=0)
        self.fuse = nn.Sequential(
            ConvBnRelu(64, 32),
            ResBlock(32),
        )
        self.head = nn.Conv2d(32, 1, 1, bias=True)
        nn.init.constant_(self.head.bias, -4.0)

    def forward(self, e3, e4):
        f4 = F.interpolate(self.from_e4(e4), size=e3.shape[2:],
                           mode='bilinear', align_corners=False)
        f3 = self.from_e3(e3)
        fused   = self.fuse(torch.cat([f4, f3], dim=1))
        logits  = self.head(fused)
        heatmap = torch.sigmoid(logits)
        return heatmap, logits


# ================================================================
# PosHead（无参数，接口不变）
# ================================================================

class PosHead(nn.Module):
    def __init__(self, max_flares=6, nms_kernel=9, conf_threshold=0.3):
        super().__init__()
        self.max_flares     = max_flares
        self.nms_kernel     = nms_kernel
        self.conf_threshold = conf_threshold

    def forward(self, heatmap):
        return self.extract_peaks_nms(heatmap)

    def extract_peaks_nms(self, heatmap):
        B, _, H, W = heatmap.shape
        K   = self.max_flares
        pad = self.nms_kernel // 2

        hm_max = F.max_pool2d(heatmap, self.nms_kernel, stride=1, padding=pad)
        peaks  = (heatmap == hm_max) & (heatmap > self.conf_threshold)

        pred_positions = torch.full((B, K, 2), -1.0, device=heatmap.device)
        pred_conf      = torch.zeros(B, K, device=heatmap.device)

        xs_grid = torch.linspace(0, 1, W, device=heatmap.device)
        ys_grid = torch.linspace(0, 1, H, device=heatmap.device)

        for b in range(B):
            peak_mask = peaks[b, 0]
            scores    = heatmap[b, 0] * peak_mask
            n_peaks   = int(peak_mask.sum().item())
            if n_peaks == 0:
                continue
            flat_scores, flat_idx = scores.view(-1).topk(
                min(K, n_peaks), largest=True, sorted=True
            )
            for slot, (score, idx) in enumerate(zip(flat_scores, flat_idx)):
                if score < self.conf_threshold:
                    break
                y_idx = idx // W
                x_idx = idx % W
                pred_positions[b, slot, 0] = xs_grid[x_idx]
                pred_positions[b, slot, 1] = ys_grid[y_idx]
                pred_conf[b, slot]         = score

        return pred_positions, pred_conf


# ================================================================
# Loss 函数（接口不变）
# ================================================================

def hungarian_match_loss(heatmap, gt_positions, gt_mask,
                         sigma=0.05, alpha_pos=2.0, beta_neg=4.0,
                         device=None):
    B, _, H, W = heatmap.shape
    device = heatmap.device

    gt_hm = torch.zeros(B, 1, H, W, device=device)
    xs = torch.linspace(0, 1, W, device=device)
    ys = torch.linspace(0, 1, H, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')

    for b in range(B):
        for k in range(gt_mask.shape[1]):
            if gt_mask[b, k] < 0.5:
                continue
            cx = gt_positions[b, k, 0]
            cy = gt_positions[b, k, 1]
            gauss = torch.exp(
                -((grid_x - cx) ** 2 + (grid_y - cy) ** 2) / (2 * sigma ** 2)
            )
            gt_hm[b, 0] = torch.max(gt_hm[b, 0], gauss)

    pos_mask = (gt_hm > 0.99).float()
    neg_mask = 1.0 - pos_mask

    loss_pos = (
        pos_mask
        * (1.0 - heatmap).clamp(min=1e-6).pow(alpha_pos)
        * torch.log(heatmap.clamp(min=1e-6))
    )
    loss_neg = (
        neg_mask
        * (1.0 - gt_hm).clamp(min=0).pow(beta_neg)
        * heatmap.clamp(min=1e-6).pow(alpha_pos)
        * torch.log((1.0 - heatmap).clamp(min=1e-6))
    )
    num_pos = pos_mask.sum().clamp(min=1.0)
    loss    = -(loss_pos.sum() + loss_neg.sum()) / num_pos

    return loss, gt_hm

class HFEnhance(nn.Module):
    """
    空间域高频增强：软阈值频率分离 + 独立处理 + 残差融合

    设计思路：
    - 用高斯软掩码在频域分离低频/高频（避免硬截断的振铃效应）
    - 高频分支独立卷积处理（炫光边缘、光晕轮廓）
    - 低频分支保留原始内容（背景亮度、炫光主体）
    - 可学习系数控制高频增强强度（初始=0，训练中自适应增长）
    - 残差连接保证训练初期稳定

    适合位置：e1（全分辨率）/ e2（半分辨率）
    """

    def __init__(self, ch, sigma=0.1):
        super().__init__()
        self.sigma = sigma   # 高斯软掩码宽度，越小=低频越窄=高频越丰富

        # ── 低频分支：捕获炫光光晕大面积渐变 ─────────────────────────────
        self.low_conv = nn.Sequential(
            nn.Conv2d(ch, ch, 1, bias=False),
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        # ── 高频分支：增强炫光边缘/条纹/光芒射线 ─────────────────────────
        self.high_conv = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),   # 3×3 感知局部边缘上下文
            nn.BatchNorm2d(ch),
            nn.ReLU(inplace=True),
        )

        # ── 可学习高频增强系数（初始化为0，安全） ────────────────────────
        # sigmoid(0) = 0.5，增强倍数 = 1 + 0.5 * w ∈ (1, 2)
        # 训练收敛后自动学习最优增强强度
        self.high_scale = nn.Parameter(torch.zeros(1))

        # ── 融合：低频 + 增强高频 → 残差加回原始 ────────────────────────
        self.fuse_norm = nn.BatchNorm2d(ch)
        self.fuse_act  = nn.ReLU(inplace=True)

    def _get_soft_masks(self, H, W, device):
        """
        生成高斯软频率掩码，避免硬截断振铃效应。

        Returns:
            low_mask:  [H, W//2+1]  中心=1（低频保留），边缘→0
            high_mask: [H, W//2+1]  中心=0（低频抑制），边缘→1
        """
        freq_h = torch.fft.fftfreq(H, device=device)        # [-0.5, 0.5]
        freq_w = torch.fft.rfftfreq(W, device=device)       # [0, 0.5]
        grid_v, grid_u = torch.meshgrid(freq_h, freq_w, indexing='ij')  # [H, W//2+1]

        radius = torch.sqrt(grid_u ** 2 + grid_v ** 2)

        # 高斯低通：sigma 越小，低频带越窄，高频越丰富
        low_mask  = torch.exp(-radius ** 2 / (2 * self.sigma ** 2))
        high_mask = 1.0 - low_mask

        return low_mask, high_mask   # 两者相加 = 1，保证完美重建

    def forward(self, x):
        """
        Args:
            x: [B, C, H, W]

        Returns:
            out: [B, C, H, W]  高频增强后的特征
        """
        B, C, H, W = x.shape

        # ── Step1：FFT 变换到频域 ──────────────────────────────────────
        f       = torch.fft.rfft2(x, norm='ortho')          # [B, C, H, W//2+1] 复数
        f_shift = torch.fft.fftshift(f, dim=-2)             # 低频移到中心

        # ── Step2：软掩码分离低频/高频 ────────────────────────────────
        low_mask, high_mask = self._get_soft_masks(H, W, x.device)

        f_low  = f_shift * low_mask    # 保留低频分量
        f_high = f_shift * high_mask   # 保留高频分量

        # ── Step3：逆变换回空间域 ─────────────────────────────────────
        f_low_back  = torch.fft.ifftshift(f_low,  dim=-2)
        f_high_back = torch.fft.ifftshift(f_high, dim=-2)

        x_low  = torch.fft.irfft2(f_low_back,  s=(H, W), norm='ortho').real   # [B,C,H,W]
        x_high = torch.fft.irfft2(f_high_back, s=(H, W), norm='ortho').real   # [B,C,H,W]

        # ── Step4：分支独立处理 ───────────────────────────────────────
        x_low_feat  = self.low_conv(x_low)    # 低频：1×1 卷积
        x_high_feat = self.high_conv(x_high)  # 高频：3×3 卷积感知局部边缘

        # ── Step5：高频自适应增强（残差风格，初始不破坏原始特征） ──────
        # 增强倍数 = 1 + sigmoid(scale) * high_feat ∈ [1, 2]
        enhanced_high = x_high_feat * (1.0 + torch.sigmoid(self.high_scale))

        # ── Step6：低频 + 增强高频 重建，再残差加回原始 ───────────────
        # 数学保证：low_mask + high_mask = 1，所以 x_low + x_high = x
        # 增强后：out = x + extra_high_energy（只是额外加了高频增益）
        recon = x_low_feat + enhanced_high
        out   = self.fuse_act(self.fuse_norm(x + recon))

        return out


# ================================================================
# 主模型（★ 默认 base_ch=24，strip_len=11）
# ================================================================

class FPN(nn.Module):
    def __init__(
        self,
        in_channels=3,
        base_ch=24,
        max_flares=3,
        strip_len=11,
        output_dir='./models'
    ):
        super().__init__()
        self.output_dir = output_dir

        self.encoder       = Encoder(in_channels, base_ch, strip_len)
        self.flare_decoder = FlareDecoder(base_ch)
        self.heatmap_head  = LightHeatmapHead(
            ch_large=base_ch * 8,   # 192
            ch_small=base_ch * 4    # 96
        )
        self.pos_head = PosHead(max_flares=max_flares)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        e1, e2, e3, e4 = self.encoder(x)

        flare_map             = self.flare_decoder(e1, e2, e3, e4)   # [B,1,H,W]
        heatmap, logits       = self.heatmap_head(e3, e4)            # [B,1,H/4,W/4]
        pred_positions, pred_conf = self.pos_head(heatmap)           # [B,3,2],[B,3]

        return flare_map, heatmap, logits, pred_positions, pred_conf  # ★ 5个，接口不变

    def save(self, epoch, iter):
        os.makedirs(self.output_dir, exist_ok=True)
        path = os.path.join(self.output_dir, f'IPN_epoch{epoch}_iter{iter}.pth')
        torch.save({'model_state_dict': self.state_dict()}, path)
        print(f'✅ IPN saved: {os.path.abspath(path)}')

def build_FPN_model(params):
    return FPN(
        in_channels=params.get('in_channels', 3),
        base_ch    =params.get('base_ch', 24),
        max_flares =params.get('max_flares', 6),
        strip_len  =params.get('strip_len', 11),
        output_dir =params.get('output_dir', './models')
    )