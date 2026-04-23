from collections import OrderedDict
from os import path as osp
import numpy as np
import torchvision

from basicsr.archs import build_network
from basicsr.losses import build_loss

from basicsr.models.sr_model import SRModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from basicsr.utils.flare_util import blend_light_source,mkdir,predict_flare_from_6_channel,predict_flare_from_3_channel
from kornia.metrics import psnr,ssim
from basicsr.metrics import calculate_metric
import torch
from tqdm import tqdm
import torch.nn.functional as torch_relu   # 文件顶部只加一次
from basicsr.models.FPN import build_FPN_model

from torchvision.transforms.functional import rgb_to_grayscale
import torch.nn.functional as F
import os
import cv2

from kornia.metrics import ssim as ssim_func




def extract_mask_standard(img_seg):
    """
    提取 Clean 区域掩码，与 evaluate_FlareX_fu.py 的 extract_mask 逻辑对齐

    注意：白色区域（值接近255）将被忽略，黑色区域（值接近0）将被计算
    返回: [H, W] 的 clean_mask，值域 0~1
    """
    # 与 extract_mask 一致：根据值域范围反转
    if img_seg.max() > 1.0:
        # 0~255 范围：白色(255)→0，黑色(0)→1
        clean_mask = 1.0 - (img_seg.astype(float) / 255.0).clip(0, 1)
    else:
        # 0~1 范围：白色(1.0)→0，黑色(0.0)→1
        clean_mask = 1.0 - img_seg.astype(float).clip(0, 1)

    clean_mask = clean_mask.clip(0, 1)

    # 统一返回 2D（调用处会处理维度）
    if clean_mask.ndim == 3:
        clean_mask = clean_mask[:, :, 0]

    return clean_mask   # [H, W]


def calculate_masked_psnr(img1, img2, mask):
    """
    计算被 mask 覆盖区域的 PSNR
    ★ 与 evaluate_FlareX_fu.py 的 compare_score + extract_mask 完全对齐
       关键：mask_area = np.sum(clean_mask_3ch) / (h*w)，与 evaluate 保持一致
    """
    from skimage.metrics import mean_squared_error as compare_mse

    # ★ 支持 [H,W] 和 [H,W,3] 两种输入
    if mask.ndim == 2:
        # 调用方传入 2D mask：复制成 3 通道
        mask_3ch = np.stack([mask, mask, mask], axis=2)   # [H, W, 3]
    else:
        # 调用方传入 [H,W,3] mask：直接使用（与 evaluate 的 clean_mask_3ch 完全一致）
        mask_3ch = mask                                     # [H, W, 3]

    h, w = mask_3ch.shape[:2]

    # ★★★ 关键修复：mask_area 用三通道 sum，与 evaluate 完全一致
    # evaluate:  mask_dict['clean'] = [np.sum(clean_mask) / (h*w), clean_mask_3ch]
    #            其中 clean_mask 是 [H,W,3]，np.sum 对所有元素求和
    mask_area = np.sum(mask_3ch) / (h * w)   # ← 原来是 np.sum(mask_2d)/(h*w)，差 3 倍！

    if mask_area <= 0:
        return 0

    img_gt_masked   = img1.astype(np.float64) * mask_3ch
    img_pred_masked = img2.astype(np.float64) * mask_3ch

    input_mse = compare_mse(img_gt_masked, img_pred_masked) / (255 * 255 * mask_area)

    if input_mse == 0:
        return float('inf')

    return 10 * np.log10((1.0 ** 2) / input_mse)

def calculate_masked_ssim(img1, img2, mask):
    """
    计算被 mask 覆盖区域的 SSIM
    先在原图上算全图 SSIM map，然后用 mask 加权求平均
    """
    from skimage.metrics import structural_similarity as ssim_skimage

    # ★ 关键修复：无论传入 [H,W] 还是 [H,W,3]，统一降为 2D
    # 防止 mask[..., None] 在 3D 输入时产生 [H,W,3,1] → repeat后变 [H,W,9,1] 的维度爆炸
    if mask.ndim == 3:
        mask = mask[:, :, 0]   # [H, W]

    # full=True 返回一张和原图尺寸一样的 SSIM 逐像素图
    _, ssim_map = ssim_skimage(img1, img2, channel_axis=2, full=True, data_range=255)
    # ssim_map: [H, W, 3]

    # 此时 mask 必然是 [H, W]，操作安全
    mask_3c = np.repeat(mask[..., None], 3, axis=2)   # [H, W, 3]

    masked_ssim_sum = np.sum(ssim_map * mask_3c)       # [H,W,3] * [H,W,3] ✅
    masked_area = np.sum(mask_3c) + 1e-8
    return masked_ssim_sum / masked_area


# ==============================================================

@MODEL_REGISTRY.register()
class DeflareModel(SRModel):
    def __init__(self, opt):
        print("DeflareModel initialization with options:", opt)
        super(DeflareModel, self).__init__(opt)

        # 测试状态标志
        self.is_testing = False
        self.current_test_iter = 0

        # ⭐ 为每次测试创建唯一的时间戳
        from datetime import datetime
        self.test_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 检测可视化配置
        base_vis_path = opt.get('val', {}).get('vis_save_path',
                                               '/home/girl/Flare/Flare7K/visualization_loctar/test')

        self.vis_config = {
            'enable': opt.get('val', {}).get('visualize_detection', False),
            'save_path': f"{base_vis_path}/{self.test_timestamp}",
            'save_all_samples': opt.get('val', {}).get('save_all_samples', False),
            'vis_interval': opt.get('val', {}).get('vis_interval', 1)
        }

        # ⭐ LPG 可视化配置
        self.lpg_vis_config = {
            'enable_train': opt.get('train', {}).get('lpg', {}).get('visualize_lpg', True),
            'enable_test': opt.get('val', {}).get('visualize_lpg', True),
            'save_path': opt.get('train', {}).get('lpg', {}).get('lpg_vis_path',
                                '/home/girl/Flare/Flare7K/lpg_visualization'),
            'train_interval': opt.get('train', {}).get('lpg', {}).get('vis_interval', 500),  # 训练时每500 iter保存
            'test_interval': 1  # 测试时每个样本都保存
        }

        print(f"📂 检测可视化保存路径: {self.vis_config['save_path']}")
        print(f"📂 LPG可视化保存路径: {self.lpg_vis_config['save_path']}")

        # 测试状态标志
        self.is_testing = False
        self.current_test_iter = 0

        # ⭐ 为每次测试创建唯一的时间戳
        from datetime import datetime
        self.test_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

        # 可视化配置
        base_vis_path = opt.get('val', {}).get('vis_save_path',
                                               '/home/girl/Flare/Flare7K/visualization_loctar/test')

        # ⭐ 在基础路径后添加时间戳
        self.vis_config = {
            'enable': opt.get('val', {}).get('visualize_detection', False),
            'save_path': f"{base_vis_path}/{self.test_timestamp}",  # ← 加时间戳
            'save_all_samples': opt.get('val', {}).get('save_all_samples', False),
            'vis_interval': opt.get('val', {}).get('vis_interval', 1)
        }

        print(f"📂 可视化保存路径: {self.vis_config['save_path']}")

    def init_training_settings(self):
        if 'train' not in self.opt:
            raise KeyError("Train options not found in model options.")

        print("Training options found:", self.opt['train'])  # 打印训练选项

        # 现在检查 'lpg' 部分是否存在
        if 'lpg' not in self.opt['train']:
            raise KeyError("LPG configuration not found in train options.")

        print("Building LPG model with params:", self.opt['train']['lpg'])  # 打印lpg配置
        # 初始化 LPG 模块

        lpg_cfg = self.opt['train']['lpg']

        # ⭐ 读取三阶段 max_sources，带默认值保证兼容
        self.max_sources_lpg_train = lpg_cfg.get('max_sources_lpg_train', 3)
        self.max_sources_main_train = lpg_cfg.get('max_sources_main_train', 1)
        self.max_sources_test = lpg_cfg.get('max_sources_test', 6)

        # 建模型时用阶段① 的 K
        lpg_cfg['max_sources'] = self.max_sources_lpg_train
        self.lpg = build_IPN_model(lpg_cfg)
        self.lpg.to(self.device)

        # ⭐⭐⭐ 加载预训练权重（关键！）
        lpg_weights_path = self.opt['train']['lpg'].get('lpg_weights', None)
        if lpg_weights_path is not None and lpg_weights_path != '~':
            print(f"🔄 正在加载 LPG 预训练权重: {lpg_weights_path}")

            # 加载权重文件
            checkpoint = torch.load(lpg_weights_path, map_location=self.device)

            # 根据保存格式加载（兼容不同保存方式）
            if 'model_state_dict' in checkpoint:
                self.lpg.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ LPG 权重加载成功 (从 model_state_dict)")
            elif 'state_dict' in checkpoint:
                self.lpg.load_state_dict(checkpoint['state_dict'])
                print(f"✅ LPG 权重加载成功 (从 state_dict)")
            else:
                # 直接是 state_dict
                self.lpg.load_state_dict(checkpoint)
                print(f"✅ LPG 权重加载成功 (直接加载)")

            # 验证权重已加载（打印第一层权重的统计信息）
            first_param = next(self.lpg.parameters())
            print(f"📊 LPG 第一层权重统计: mean={first_param.mean().item():.6f}, "
                  f"std={first_param.std().item():.6f}, "
                  f"min={first_param.min().item():.6f}, "
                  f"max={first_param.max().item():.6f}")
        else:
            print("⚠️  未指定 LPG 预训练权重，使用随机初始化")

        # ⭐ 如果需要冻结 LPG
        if self.opt['train']['lpg'].get('freeze_lpg', False):
            print("🔒 冻结 LPG 参数")
            for param in self.lpg.parameters():
                param.requires_grad = False
            self.lpg.eval()  # 设置为评估模式
        else:
            print("🔓 LPG 参数可训练")
            self.lpg.train()

        self.net_g.train()
        train_opt = self.opt['train']
        self.output_ch=self.opt['network_g']['output_ch']
        if 'multi_stage' in self.opt['network_g']:
            self.multi_stage=self.opt['network_g']['multi_stage']
        else:
            self.multi_stage=1
        print("Output channel is:", self.output_ch)
        print("Network contains",self.multi_stage,"stages.")

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = build_network(self.opt['network_g']).to(self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path, self.opt['path'].get('strict_load_g', True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        self.l1_pix = build_loss(train_opt['l1_opt']).to(self.device)



        self.l_perceptual = build_loss(train_opt['perceptual']).to(self.device)

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        self.gt = data['gt'].to(self.device)

        if 'flare' in data:
            self.flare = data['flare'].to(self.device)
            self.gamma = data['gamma']

        if 'mask' in data:
            self.mask = data['mask'].to(self.device)

        if 'light' in data:
            self.light = data['light'].to(self.device)

        if 'light_and_flare' in data:
            self.light_and_flare = data['light_and_flare'].to(self.device)

        if 'background' in data:
            self.background = data['background'].to(self.device)

        if 'light_positions' in data:
            self.light_positions = data['light_positions'].to(self.device)  # [B, 3, 2]

        if 'light_pos_mask' in data:
            self.light_pos_mask = data['light_pos_mask'].to(self.device)  # [B, 3]

        # ★ heatmap_gt 不再依赖 dataset，由 loss 内部从坐标渲染，保留兼容即可
        if 'heatmap_gt' in data:
            self.heatmap_gt = data['heatmap_gt'].to(self.device)

        if 'num_flares' in data:
            self.num_flares = data['num_flares']

    def _render_pos_map(self,
                        pred_positions: torch.Tensor,
                        pred_conf: torch.Tensor,
                        H: int,
                        W: int,
                        sigma: float = 0.2,
                        conf_thresh: float = 0.1
                        ) -> torch.Tensor:
        """
        将 IPN 输出的归一化坐标渲染成高斯热点图。

        Args:
            pred_positions: [B, K, 2]  归一化坐标 (x, y) ∈ [0, 1]
            pred_conf:      [B, K]     各光源置信度
            H, W:           目标图像尺寸
            sigma:          高斯半径（归一化坐标），约图像宽度的 3%
            conf_thresh:    低于此置信度的点不渲染

        Returns:
            pos_map: [B, 1, H, W]  高斯热点图，值域 [0, 1]
        """
        B, K, _ = pred_positions.shape
        device = pred_positions.device

        # 归一化坐标网格
        xs = torch.linspace(0, 1, W, device=device)  # [W]
        ys = torch.linspace(0, 1, H, device=device)  # [H]
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

        pos_map = torch.zeros(B, 1, H, W, device=device)

        for b in range(B):
            for k in range(K):
                # 置信度过滤
                if float(pred_conf[b, k]) < conf_thresh:
                    continue

                cx = pred_positions[b, k, 0]  # 归一化 x
                cy = pred_positions[b, k, 1]  # 归一化 y

                # 高斯核
                gauss = torch.exp(
                    -((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                    / (2 * sigma ** 2)
                )  # [H, W]

                # 多峰叠加取 max
                pos_map[b, 0] = torch.max(pos_map[b, 0], gauss)

        return pos_map.clamp(0.0, 1.0)

    def _set_lpg_max_sources(self, k: int):
        """动态修改 IPN 的最大光源数，递归覆盖所有子模块中的同名属性。"""
        for _, module in self.lpg.named_modules():
            if hasattr(module, 'max_sources') and module.max_sources != k:
                module.max_sources = k
        # 顶层兜底
        if hasattr(self.lpg, 'max_sources'):
            self.lpg.max_sources = k

    def optimize_parameters(self, current_iter):
        if current_iter <= self.opt['train']['lpg']['lpg_iter']:
            self._set_lpg_max_sources(self.max_sources_lpg_train)
            self.optimizer_lpg.zero_grad()

            # ── forward：5 个返回值 ──────────────────────────────────
            flare_map, heatmap, logits, pred_positions, pred_conf = self.lpg(self.lq)
            # flare_map:      [B, 1, H, W]
            # heatmap:        [B, 1, H/4, W/4]  已过 sigmoid
            # logits:         [B, 1, H/4, W/4]  sigmoid 之前的原始值
            # pred_positions: [B, 3, 2]         训练阶段仅供可视化
            # pred_conf:      [B, 3]            训练阶段仅供可视化

            # ── GT 预处理 ────────────────────────────────────────────
            self.light_and_flare_gray = \
                torchvision.transforms.functional.rgb_to_grayscale(
                    self.light_and_flare, num_output_channels=1
                )  # [B, 1, H, W]

            # ============================================================
            # Loss 1：炫光强度回归损失
            # ============================================================
            seg_gt = self.light_and_flare_gray  # [B, 1, H, W]，连续值 0~1

            # ── 1.1 Charbonnier L1（主力回归损失）───────────────────────
            # 直接优化每个像素的数值误差，比 BCE 更适合连续强度值
            loss_intensity = torch.mean(
                torch.sqrt((flare_map - seg_gt) ** 2 + 1e-6)
            )

            # ── 1.2 修正 Focal Loss（去掉二值化，改为误差驱动权重）───────
            # 原来：pt = where(seg_gt > 0.1, flare_map, 1-flare_map)  ← 弱炫光被当背景
            # 现在：误差越大 → weight 越大，强/弱炫光都会贡献梯度
            bce_per_pixel = F.binary_cross_entropy(flare_map, seg_gt, reduction='none')
            error_weight = (flare_map.detach() - seg_gt).abs().clamp(min=0.02)  # 误差越大越关注
            loss_seg_focal = (error_weight * bce_per_pixel).mean()

            # ── 1.3 弱炫光专项损失（直接修复归零问题）────────────────────
            # 专门对 seg_gt ∈ (0.02, 0.35) 的弱炫光区域加强惩罚
            weak_mask = ((seg_gt > 0.02) & (seg_gt < 0.35)).float()  # 弱炫光区域
            weak_pixel_count = weak_mask.sum().clamp(min=1.0)
            loss_weak = (weak_mask * (flare_map - seg_gt).abs()).sum() / weak_pixel_count

            # ── 合并（去掉 Dice Loss）────────────────────────────────────
            loss_seg = (
                    2.0 * loss_intensity  # 主力：全局强度贴合
                    + 0.5 * loss_seg_focal  # 辅助：难例关注
                    + 2.0 * loss_weak  # 专项：弱炫光不归零
            )



            # ============================================================
            # Loss 2：热力图损失（CornerNet Focal Loss + 高斯GT渲染）
            # ★ GT 热力图在 loss 内部从坐标渲染，不依赖 dataset
            # ★ 使用 logits 做 Focal Loss，梯度比 sigmoid 后更强
            # ============================================================
            loss_heatmap = torch.tensor(0.0, device=self.device)
            rendered_gt_hm = None  # 供可视化用

            if hasattr(self, 'light_positions') and hasattr(self, 'light_pos_mask'):
                B, _, Hh, Wh = heatmap.shape  # Hh=H/4, Wh=W/4

                # ── 渲染 GT 热力图（高斯核叠加）────────────────────────
                gt_hm = torch.zeros(B, 1, Hh, Wh, device=self.device)
                sigma = 0.05  # 归一化坐标下的高斯宽度（≈图像5%宽度）

                xs = torch.linspace(0, 1, Wh, device=self.device)
                ys = torch.linspace(0, 1, Hh, device=self.device)
                grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [Hh, Wh]

                for b in range(B):
                    for k in range(self.light_pos_mask.shape[1]):
                        if self.light_pos_mask[b, k] < 0.5:
                            continue
                        cx = self.light_positions[b, k, 0]
                        cy = self.light_positions[b, k, 1]
                        gauss = torch.exp(
                            -((grid_x - cx) ** 2 + (grid_y - cy) ** 2)
                            / (2 * sigma ** 2)
                        )
                        gt_hm[b, 0] = torch.max(gt_hm[b, 0], gauss)  # 多峰取 max

                rendered_gt_hm = gt_hm.detach()  # 供可视化，不参与梯度

                # ── CornerNet Focal Loss（直接作用在 logits 上）─────────
                # 正样本（峰值处）：(1 - sigmoid(logit))^2 * log(sigmoid(logit))
                # 负样本：(1 - gt_hm)^4 * sigmoid(logit)^2 * log(1 - sigmoid(logit))
                alpha_hm = 2.0
                beta_hm = 4.0

                pos_mask_hm = (gt_hm > 0.99).float()
                neg_mask_hm = 1.0 - pos_mask_hm

                loss_hm_pos = (
                        pos_mask_hm
                        * (1.0 - heatmap).clamp(min=1e-6).pow(alpha_hm)
                        * torch.log(heatmap.clamp(min=1e-6))
                )
                loss_hm_neg = (
                        neg_mask_hm
                        * (1.0 - gt_hm).clamp(min=0).pow(beta_hm)
                        * heatmap.clamp(min=1e-6).pow(alpha_hm)
                        * torch.log((1.0 - heatmap).clamp(min=1e-6))
                )
                num_pos_hm = pos_mask_hm.sum().clamp(min=1.0)
                loss_heatmap = -(loss_hm_pos.sum() + loss_hm_neg.sum()) / num_pos_hm

            # ============================================================
            # 总损失（去掉 loss_pos 和 loss_conf，坐标由热力图隐式学习）
            # ============================================================
            w_hm = 5.0
            total_loss = loss_seg + w_hm * loss_heatmap
            total_loss.backward()

            # ============================================================
            # 可视化（每500步）
            # ============================================================
            # ============================================================
            # 可视化（每500步）
            # ============================================================
            # ============================================================
            # 可视化（每500步）
            # ============================================================
            if current_iter % 500 == 0:
                save_root = '/home/girl/Flare/Flare7K/data_flare+core_visulal'
                os.makedirs(save_root, exist_ok=True)
                H, W = self.lq.shape[2], self.lq.shape[3]
                _, _, Hh, Wh = heatmap.shape

                # ══════════════════════════════════════════════════════
                # 工具函数
                # ══════════════════════════════════════════════════════
                def to_bgr(t):
                    """RGB tensor [C,H,W] → BGR ndarray [H,W,3]"""
                    np_img = (t.clamp(0, 1) * 255).byte().cpu().numpy()
                    return np_img.transpose(1, 2, 0)[:, :, ::-1].copy()

                def to_gray3ch(t):
                    """单通道 tensor [1,H,W] → 3通道灰度 BGR ndarray [H,W,3]"""
                    gray = (t[0].clamp(0, 1) * 255).byte().cpu().numpy()
                    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

                def to_jet(t):
                    """单通道 tensor [1,H,W] → JET 伪彩色 BGR ndarray [H,W,3]"""
                    np_gray = (t[0].clamp(0, 1).cpu().numpy() * 255).astype('uint8')
                    return cv2.applyColorMap(np_gray, cv2.COLORMAP_JET)

                def to_jet_resized(t, target_h, target_w):
                    """单通道 tensor [1,Hh,Wh] → JET 伪彩色，resize 到 (target_h, target_w)"""
                    jet = to_jet(t)
                    return cv2.resize(jet, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

                def overlay_jet(base_bgr, jet_bgr, alpha=0.45):
                    """把 jet_bgr 半透明叠加到 base_bgr 上，两者必须同尺寸"""
                    return cv2.addWeighted(base_bgr, 1.0 - alpha, jet_bgr, alpha, 0)

                def diff_map_bgr(pred_t, gt_t, amplify=4.0):
                    """
                    计算 |pred - gt| 差值图 → [H,W,3] BGR
                    pred_t, gt_t: [1,H,W] 单通道 tensor
                    ★ 修复：squeeze后转灰度3通道，避免 [H,W,1] 导致 vstack 通道不匹配
                    """
                    diff = (pred_t.detach() - gt_t).abs().clamp(0, 1)
                    diff_amp = (diff * amplify).clamp(0, 1)
                    # [1,H,W] → [H,W] numpy
                    gray_np = (diff_amp[0].cpu().numpy() * 255).astype('uint8')
                    # 用 JET 伪彩色让差异更直观（也可以换成灰度3通道）
                    return cv2.applyColorMap(gray_np, cv2.COLORMAP_HOT)

                def add_title(img, title, bar_h=28):
                    """在图片顶部加黑色标题栏，img 必须是 [H,W,3]"""
                    assert img.ndim == 3 and img.shape[2] == 3, \
                        f"add_title 收到非3通道图像: shape={img.shape}, title={title}"
                    bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8)
                    cv2.putText(bar, title, (5, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    return np.vstack([bar, img])

                def add_stats_bar(img, text, bar_h=22):
                    """在图片底部加统计信息栏"""
                    bar = np.zeros((bar_h, img.shape[1], 3), dtype=np.uint8)
                    cv2.putText(bar, text, (5, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                                (0, 255, 128), 1, cv2.LINE_AA)
                    return np.vstack([img, bar])

                def make_row(panels):
                    """所有 panel 统一到第一张的高度，然后横向拼接"""
                    ref_h = panels[0].shape[0]
                    resized = []
                    for p in panels:
                        new_w = max(1, int(p.shape[1] * ref_h / p.shape[0]))
                        resized.append(cv2.resize(p, (new_w, ref_h)))
                    return np.hstack(resized)

                def fit_width(img, target_w):
                    """等比缩放到指定宽度"""
                    target_h = max(1, int(img.shape[0] * target_w / img.shape[1]))
                    return cv2.resize(img, (target_w, target_h))

                def row_divider(w, label):
                    """生成行间分隔条"""
                    bar = np.full((22, w, 3), 35, dtype=np.uint8)
                    cv2.putText(bar, label, (8, 16),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                                (80, 200, 80), 1, cv2.LINE_AA)
                    return bar

                def draw_points_on(base_bgr, positions_np, mask_np, color_list, is_gt=True):
                    """在底图上绘制 GT 或预测点位，返回 [H,W,3]"""
                    vis = base_bgr.copy()
                    for k in range(len(mask_np)):
                        if mask_np[k] < 0.5:
                            continue
                        xn, yn = float(positions_np[k, 0]), float(positions_np[k, 1])
                        xp, yp = int(xn * W), int(yn * H)
                        color = color_list[k % len(color_list)]
                        if is_gt:
                            cv2.circle(vis, (xp, yp), 18, (255, 255, 255), 3)
                            cv2.circle(vis, (xp, yp), 11, color, -1)
                            label = f'GT{k + 1}({xn:.2f},{yn:.2f})'
                        else:
                            cv2.circle(vis, (xp, yp), 18, (255, 255, 255), 3)
                            cv2.drawMarker(vis, (xp, yp), color,
                                           cv2.MARKER_TILTED_CROSS, 22, 3, cv2.LINE_AA)
                            label = f'P{k + 1}({xn:.2f},{yn:.2f})'
                        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            cv2.putText(vis, label, (xp + 20 + ddx, yp + 7 + ddy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.46,
                                        (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(vis, label, (xp + 20, yp + 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.46, color, 1, cv2.LINE_AA)
                    return vis

                def nms_peaks(hm_tensor, nms_k=9, thresh=0.3, max_peaks=3):
                    """hm_tensor: [1, Hh, Wh] → List[(xn, yn, conf)]"""
                    hm = hm_tensor[0]
                    pad = nms_k // 2
                    hm_4d = hm.unsqueeze(0).unsqueeze(0)
                    hm_max = F.max_pool2d(hm_4d, nms_k, stride=1, padding=pad)[0, 0]
                    peak_mask = (hm == hm_max) & (hm > thresh)
                    coords = peak_mask.nonzero(as_tuple=False)
                    if coords.shape[0] == 0:
                        return []
                    scores = hm[coords[:, 0], coords[:, 1]]
                    topk_idx = scores.topk(min(max_peaks, scores.shape[0])).indices
                    xs_grid = torch.linspace(0, 1, Wh, device=hm.device)
                    ys_grid = torch.linspace(0, 1, Hh, device=hm.device)
                    peaks = []
                    for idx in topk_idx:
                        yi = coords[idx, 0].item()
                        xi = coords[idx, 1].item()
                        peaks.append((xs_grid[xi].item(), ys_grid[yi].item(), scores[idx].item()))
                    return peaks

                COLORS = [(0, 255, 0), (0, 128, 255), (255, 0, 128)]

                # ══════════════════════════════════════════════════════
                # 基础图像准备（全部保证 [H,W,3]）
                # ══════════════════════════════════════════════════════
                lq_bgr = to_bgr(self.lq[0])  # [H,W,3] 带炫光输入
                gt_color_bgr = to_bgr(self.light_and_flare[0])  # [H,W,3] GT炫光层彩色
                seg_gt_gray = to_gray3ch(self.light_and_flare_gray[0])  # [H,W,3] GT灰度分割
                fm_gray_bgr = to_gray3ch(flare_map[0].detach())  # [H,W,3] 预测FlareMap灰度
                fm_jet_bgr = to_jet(flare_map[0].detach())  # [H,W,3] 预测FlareMap JET

                # 统计信息字符串
                fm_np = flare_map[0, 0].detach().cpu().numpy()
                seg_np = self.light_and_flare_gray[0, 0].cpu().numpy()
                fm_stats = f'min={fm_np.min():.3f} max={fm_np.max():.3f} mean={fm_np.mean():.3f}'
                seg_stats = f'min={seg_np.min():.3f} max={seg_np.max():.3f} mean={seg_np.mean():.3f}'

                # ★ 差值图（修复：用 diff_map_bgr 确保输出 [H,W,3]）
                fm_diff_bgr = diff_map_bgr(flare_map[0].detach(),
                                           self.light_and_flare_gray[0], amplify=4.0)

                # FlareMap JET 叠加到输入图上
                fm_overlay_bgr = overlay_jet(lq_bgr, fm_jet_bgr, alpha=0.5)

                # 热力图
                hm_pred_bgr = to_jet_resized(heatmap[0].detach(), H, W)  # [H,W,3]
                if rendered_gt_hm is not None:
                    hm_gt_bgr = to_jet_resized(rendered_gt_hm[0], H, W)  # [H,W,3]
                else:
                    hm_gt_bgr = np.zeros((H, W, 3), dtype=np.uint8)

                hm_pred_overlay = overlay_jet(lq_bgr, hm_pred_bgr, alpha=0.45)
                hm_gt_overlay = overlay_jet(lq_bgr, hm_gt_bgr, alpha=0.45)

                # 点位可视化
                gt_pos_bgr = lq_bgr.copy()
                pred_pos_bgr = lq_bgr.copy()
                gt_hm_pts = hm_gt_overlay.copy()
                pred_hm_pts = hm_pred_overlay.copy()
                peaks = []

                if hasattr(self, 'light_positions') and hasattr(self, 'light_pos_mask'):
                    gt_positions_np = self.light_positions[0].cpu().numpy()  # [3,2]
                    gt_mask_np = self.light_pos_mask[0].cpu().numpy()  # [3]

                    gt_pos_bgr = draw_points_on(lq_bgr, gt_positions_np,
                                                gt_mask_np, COLORS, is_gt=True)
                    gt_hm_pts = draw_points_on(hm_gt_overlay, gt_positions_np,
                                               gt_mask_np, COLORS, is_gt=True)

                    peaks = nms_peaks(heatmap[0].detach(), nms_k=9, thresh=0.3,
                                      max_peaks=self.max_sources_lpg_train)
                    pred_pos_bgr = lq_bgr.copy()
                    for ki, (xn, yn, conf) in enumerate(peaks):
                        xp, yp = int(xn * W), int(yn * H)
                        color = COLORS[ki % len(COLORS)]
                        cv2.circle(pred_pos_bgr, (xp, yp), 18, (255, 255, 255), 3)
                        cv2.drawMarker(pred_pos_bgr, (xp, yp), color,
                                       cv2.MARKER_TILTED_CROSS, 22, 3, cv2.LINE_AA)
                        label = f'P{ki + 1}({xn:.2f},{yn:.2f}) c={conf:.2f}'
                        for ddx, ddy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                            cv2.putText(pred_pos_bgr, label, (xp + 20 + ddx, yp + 7 + ddy),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.44,
                                        (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(pred_pos_bgr, label, (xp + 20, yp + 7),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, color, 1, cv2.LINE_AA)

                    pred_hm_pts = pred_pos_bgr  # 预测点也画在热力图叠加上
                    pred_hm_pts = hm_pred_overlay.copy()
                    for ki, (xn, yn, conf) in enumerate(peaks):
                        xp, yp = int(xn * W), int(yn * H)
                        color = COLORS[ki % len(COLORS)]
                        cv2.circle(pred_hm_pts, (xp, yp), 18, (255, 255, 255), 3)
                        cv2.drawMarker(pred_hm_pts, (xp, yp), color,
                                       cv2.MARKER_TILTED_CROSS, 22, 3, cv2.LINE_AA)

                # ══════════════════════════════════════════════════════
                # 拼图布局（3行 × 4列）
                # ══════════════════════════════════════════════════════
                # Row1：输入 & GT 原始数据
                row1 = [
                    add_title(lq_bgr,
                              '1.Input LQ'),
                    add_title(gt_color_bgr,
                              '2.GT light_and_flare (color)'),
                    add_title(add_stats_bar(seg_gt_gray, seg_stats),
                              '3.GT Seg Gray'),
                    add_title(gt_pos_bgr,
                              '4.GT Light Positions'),
                ]

                # Row2：FlareMap 强度分析
                row2 = [
                    add_title(add_stats_bar(fm_gray_bgr, fm_stats),
                              '5.Pred FlareMap (gray)'),
                    add_title(add_stats_bar(fm_jet_bgr, fm_stats),
                              '6.Pred FlareMap (JET)'),
                    add_title(fm_overlay_bgr,
                              '7.LQ + FlareMap overlay'),
                    add_title(fm_diff_bgr,
                              '8.Diff |Pred-GT| x4 (HOT)'),
                ]

                # Row3：热力图 & 光源定位对比
                row3 = [
                    add_title(gt_hm_pts,
                              '9.GT Heatmap + GT Pts'),
                    add_title(pred_hm_pts,
                              f'10.Pred Heatmap + NMS({len(peaks)})'),
                    add_title(gt_pos_bgr,
                              '11.GT Positions on LQ'),
                    add_title(pred_pos_bgr,
                              f'12.NMS Pred Positions({len(peaks)})'),
                ]

                r1 = make_row(row1)
                r2 = make_row(row2)
                r3 = make_row(row3)
                ref_w = r1.shape[1]

                loss_info = (f'iter={current_iter}  '
                             f'loss_seg={loss_seg.item():.4f}  '
                             f'loss_seg_focal={loss_seg_focal.item():.4f}  '
                             f'loss_hm={loss_heatmap.item():.4f}  '
                             f'total={total_loss.item():.4f}')

                overview = np.vstack([
                    row_divider(ref_w, f'[LPG Train] {loss_info}'),
                    r1,
                    row_divider(ref_w, '── FlareMap Intensity Analysis ──'),
                    fit_width(r2, ref_w),
                    row_divider(ref_w, '── Heatmap & Light Source Localization ──'),
                    fit_width(r3, ref_w),
                ])

                prefix = os.path.join(save_root, f'iter_{current_iter:06d}')
                cv2.imwrite(f'{prefix}_overview.png', overview)

                # ══════════════════════════════════════════════════════
                # 控制台输出
                # ══════════════════════════════════════════════════════
                n_gt = int(self.light_pos_mask[0].sum().item()) \
                    if hasattr(self, 'light_pos_mask') else 0
                print(f'\n{"=" * 65}')
                print(f'[iter {current_iter}]  {loss_info}')
                print(f'  FlareMap → {fm_stats}')
                print(f'  SegGT    → {seg_stats}')
                print(f'  GT光源数={n_gt}  NMS峰值数={len(peaks)}')
                if hasattr(self, 'light_positions'):
                    for k in range(n_gt):
                        gx = self.light_positions[0, k, 0].item()
                        gy = self.light_positions[0, k, 1].item()
                        print(f'    GT{k + 1}: ({gx:.3f}, {gy:.3f})')
                for ki, (xn, yn, conf) in enumerate(peaks):
                    print(f'    NMS{ki + 1}: ({xn:.3f}, {yn:.3f})  conf={conf:.3f}')
                print(f'  已保存 → {prefix}_overview.png')
                print(f'{"=" * 65}\n')

            # ── 参数更新 ──────────────────────────────────────────────────
            self.optimizer_lpg.step()

            # ── 日志 ──────────────────────────────────────────────────────
            loss_dict = {
                'l_total': total_loss,
                'l_seg': loss_seg,
                'l_intensity': loss_intensity,  # ← 新增，监控主力回归
                'l_seg_focal': loss_seg_focal,
                'l_weak': loss_weak,  # ← 新增，监控弱炫光修复效果
                'l_heatmap': loss_heatmap,
            }
            self.log_dict = self.reduce_loss_dict(loss_dict)

            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)
        else:
            self._set_lpg_max_sources(self.max_sources_main_train)
            self.optimizer_g.zero_grad()
            if hasattr(self, 'lpg') and self.opt['train']['lpg'].get('train_lpg', False):
                with torch.no_grad():
                    # ── ① IPN 推理，冻结梯度 ──────────────────────────────────
                    flare_map, heatmap, logits, pred_positions, pred_conf = self.lpg(self.lq)
                    # flare_map:      [B, 1, H,   W  ]  炫光图，已是原始尺寸
                    # heatmap:        [B, 1, H/4, W/4]  热力图，需要上采样
                    # pred_positions: [B, K, 2]          归一化坐标 (x,y) ∈ [0,1]
                    # pred_conf:      [B, K]             各光源置信度
                    # ★ 强制截断到 max_sources_main_train，不管 IPN 内部输出几个
                    K_limit = self.max_sources_main_train
                    pred_positions = pred_positions[:, :K_limit, :]  # [B, 1, 2]
                    pred_conf = pred_conf[:, :K_limit]  # [B, 1]

                    H, W = self.lq.shape[2], self.lq.shape[3]

                    # ── ② heatmap 上采样回原始分辨率 ─────────────────────────
                    heatmap_full = F.interpolate(
                        heatmap,
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )  # [B, 1, H, W]

                    # ── ③ 光源位置渲染成高斯热点图 [B, 1, H, W] ─────────────
                    pos_map = self._render_pos_map(
                        pred_positions, pred_conf,
                        H, W,
                        sigma=0.03,  # 高斯半径，约图像宽度的3%
                        conf_thresh=0.1  # 低于此置信度的点不渲染
                    )  # [B, 1, H, W]

                    # ── ④ 拼接所有信息作为主干网络输入 ──────────────────────
                    # lq:           [B, 3, H, W]  原始带炫光图
                    # flare_map:    [B, 1, H, W]  炫光区域掩码
                    # heatmap_full: [B, 1, H, W]  光源热力图（上采样）
                    # pos_map:      [B, 1, H, W]  光源位置高斯图
                    net_input = torch.cat(
                        [self.lq, flare_map, heatmap_full, pos_map], dim=1
                    )  # [B, 6, H, W]

            else:
                net_input = self.lq  # baseline 模式，输入通道数=3

            self.output = self.net_g(net_input,iter_idx=current_iter, vis_save_path="./experiment/vis_logs")

            # 处理输出通道，根据不同通道数处理不同逻辑
            if self.output_ch == 6:
                self.deflare, self.flare_hat, self.merge_hat = predict_flare_from_6_channel(self.output, self.gamma)
            elif self.output_ch == 3:
                self.mask = torch.zeros_like(self.lq).cuda()  # 根据需求使用掩码
                self.deflare, self.flare_hat = predict_flare_from_3_channel(
                    self.output,
                    self.mask,
                    self.lq,
                    self.flare,
                    self.lq,
                    self.gamma
                )
            else:
                raise ValueError("Output channel should be either 3 or 6.")

            # 初始化总损失
            total_loss = 0
            loss_dict = OrderedDict()

            # 计算 L1 损失
            l1_flare = self.l1_pix(self.flare_hat, self.flare)
            loss_dict['l1_flare'] = l1_flare
            total_loss += l1_flare


            l1_base = self.l1_pix(self.deflare, self.gt)
            loss_dict['l1_base'] = l1_base
            total_loss += l1_base

            if self.output_ch == 6:
                # 如果是六通道, 计算重建损失
                l1_recons = self.l1_pix(self.merge_hat, self.lq)
                loss_dict['l1_recons'] = l1_recons * 2
                total_loss += l1_recons * 2


            # 计算感知损失
            l_vgg_flare = self.l_perceptual(self.flare_hat, self.flare)
            l_vgg_base = self.l_perceptual(self.deflare, self.gt)
            l_vgg = l_vgg_base + l_vgg_flare
            #3.18-fzj
            loss_dict['l_vgg'] = l_vgg
            loss_dict['l_vgg_base'] = l_vgg_base
            loss_dict['l_vgg_flare'] = l_vgg_flare

            total_loss += l_vgg

            # 反向传播
            total_loss.backward()

            # 更新优化器
            self.optimizer_g.step()

            # 清理显存：根据需要在每10个迭代中清理
            if current_iter % 10 == 0:
                torch.cuda.empty_cache()

            # 记录损失
            self.log_dict = self.reduce_loss_dict(loss_dict)

            # 如果使用 EMA 则更新
            if self.ema_decay > 0:
                self.model_ema(decay=self.ema_decay)

    def test(self):
        """测试方法"""

        if hasattr(self, 'lpg') and self.opt['train']['lpg'].get('train_lpg', False):
            self._set_lpg_max_sources(self.max_sources_test)
            self.net_g_ema.eval()
            with torch.no_grad():
                # ── ① LPG 推理，K=6 ──────────────────────────────────────
                flare_map, heatmap, logits, pred_positions, pred_conf = self.lpg(self.lq)

                H, W = self.lq.shape[2], self.lq.shape[3]

                # ── ② heatmap 上采样回原始分辨率 ─────────────────────────
                heatmap_full = F.interpolate(
                    heatmap, size=(H, W),
                    mode='bilinear', align_corners=False
                )  # [B, 1, H, W]

                # ── ③ 光源位置渲染成高斯热点图 ───────────────────────────
                pos_map = self._render_pos_map(
                    pred_positions, pred_conf,
                    H, W,
                    sigma=0.1,
                    conf_thresh=0.1
                )  # [B, 1, H, W]

                # ── ④ 拼接 6 通道输入 ────────────────────────────────────
                net_input = torch.cat(
                    [self.lq, flare_map, heatmap_full, pos_map], dim=1
                )  # [B, 6, H, W]

                # ── ⑤ 主干网络推理 ───────────────────────────────────────
                if self.vis_config['enable']:
                    self.output = self.net_g_ema(
                        net_input,
                        iter_idx=self.current_test_iter,
                        enable_vis=True,
                        vis_save_path=self.vis_config['save_path']
                    )
                else:
                    self.output = self.net_g_ema(net_input)

        else:
            self.net_g.eval()
            with torch.no_grad():
                if self.vis_config['enable']:
                    self.output = self.net_g(
                        self.lq,
                        iter_idx=self.current_test_iter,
                        enable_vis=True,
                        vis_save_path=self.vis_config['save_path']
                    )
                else:
                    self.output = self.net_g(self.lq)

        # ── ⑥ 处理输出通道 ───────────────────────────────────────────────
        if self.output_ch == 6:
            self.deflare, self.flare_hat, self.merge_hat = predict_flare_from_6_channel(
                self.output, self.gamma
            )
        elif self.output_ch == 3:
            self.mask = torch.zeros_like(self.lq).cuda()
            self.deflare, self.flare_hat = predict_flare_from_3_channel(
                self.output, self.mask, self.gt, self.flare, self.lq, self.gamma
            )
        else:
            assert False, "Error! Output channel should be defined as 3 or 6."

        if not hasattr(self, 'net_g_ema'):
            self.net_g.train()


    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        """验证方法"""
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        # 判断当前是否是第二个用于评估 clean 区域的测试集
        is_val_2 = 'clean' in dataset_name.lower() or 'val_2' in dataset_name.lower()

        if with_metrics:
            if is_val_2:
                self.metric_results = {'clean_psnr': 0, 'clean_ssim': 0}
            else:
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}

            self._initialize_best_metric_results(dataset_name)

            if is_val_2:
                for m in ['clean_psnr', 'clean_ssim']:
                    if m not in self.best_metric_results[dataset_name]:
                        self.best_metric_results[dataset_name][m] = {
                            'better': 'higher',
                            'val': float('-inf'),
                            'iter': -1
                        }

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        # ══════════════════════════════════════════════════════════════
        # 构建 mask 文件索引：排序列表 + 文件名→路径字典（双保险）
        # Image_Pair_Loader 不加载 mask，必须直接从磁盘读取
        # ══════════════════════════════════════════════════════════════
        import glob as _glob
        _mask_file_list = []  # 按文件名排序的列表（用于 idx 回退匹配）
        _mask_name_dict = {}  # 文件名(无扩展名) → 完整路径（用于按名精确匹配）
        if is_val_2:
            _mask_root = dataloader.dataset.opt.get('dataroot_mask', None)
            if _mask_root and os.path.isdir(_mask_root):
                for _ext in ['*.png', '*.jpg', '*.bmp', '*.PNG', '*.JPG', '*.jpeg']:
                    _mask_file_list.extend(_glob.glob(os.path.join(_mask_root, _ext)))
                _mask_file_list = sorted(_mask_file_list)
                for _p in _mask_file_list:
                    _stem = os.path.splitext(os.path.basename(_p))[0]
                    _mask_name_dict[_stem] = _p
                # print(f"[Mask] 从 {_mask_root} 找到 {len(_mask_file_list)} 个 mask 文件")

            else:
                print(f"[Mask] ⚠️ dataroot_mask 未配置或路径不存在: {_mask_root}")

        # ══════════════════════════════════════════════════════════════
        # 主循环
        # ══════════════════════════════════════════════════════════════
        _valid_mask_count = 0  # 记录成功配对 mask 的图像数量

        for idx, val_data in enumerate(dataloader):
            self.current_test_iter = idx

            # ── 调试：第一张时打印 val_data 的所有键，确认路径字段名 ──

            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img

            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # ══════════════════════════════════════════════════════════
            # is_val_2：计算 clean 区域的 PSNR / SSIM
            # ══════════════════════════════════════════════════════════
            if is_val_2:
                clean_mask = None
                _used_mask_path = None

                # ── Step 1：尝试从 val_data 获取当前图像文件名，精确匹配 mask ──
                _img_path = None
                for _path_key in ['lq_path', 'lq_paths', 'gt_path', 'gt_paths',
                                  'img_path', 'path', 'lq_name', 'gt_name']:
                    _raw = val_data.get(_path_key, None)
                    if _raw is None:
                        continue
                    _candidate = _raw[0] if isinstance(_raw, (list, tuple)) else _raw
                    if isinstance(_candidate, torch.Tensor):
                        continue
                    _img_path = str(_candidate)
                    break

                if _img_path is not None and _mask_name_dict:
                    _img_stem = os.path.splitext(os.path.basename(_img_path))[0]
                    # 精确匹配
                    if _img_stem in _mask_name_dict:
                        _used_mask_path = _mask_name_dict[_img_stem]
                    else:
                        # 模糊匹配（mask 文件名含图像名，或反之）
                        for _mk, _mv in _mask_name_dict.items():
                            if _img_stem in _mk or _mk in _img_stem:
                                _used_mask_path = _mv
                                break

                # ── Step 2：精确匹配失败 → 按 idx 回退（必须保证 dataloader 与 mask 同序）──
                if _used_mask_path is None and _mask_file_list and idx < len(_mask_file_list):
                    _used_mask_path = _mask_file_list[idx]

                # ── Step 3：读取 mask 文件 ────────────────────────────────────
                if _used_mask_path is not None:
                    _mask_bgr = cv2.imread(_used_mask_path)
                    if _mask_bgr is not None:
                        _mask_rgb = cv2.cvtColor(_mask_bgr, cv2.COLOR_BGR2RGB)  # [H, W, 3] RGB

                        # ★★★ 关键修复：与 evaluate 的 extract_mask 完全一致
                        # evaluate:  clean_mask = 1.0 - (img_seg / 255.0).clip(0, 1)  → [H, W, 3]
                        # 保留三通道，不再调用 extract_mask_standard（其内部会 [:,:,0] 降维，导致差 3 倍）
                        clean_mask = 1.0 - (_mask_rgb.astype(float) / 255.0).clip(0, 1)  # [H, W, 3]
                        clean_mask = clean_mask.clip(0, 1)  # [H, W, 3]

                        _valid_mask_count += 1

                        if idx < 5 or idx % 20 == 0:
                            _img_name_dbg = os.path.basename(_img_path) if _img_path else f'idx={idx}'
                            # ★ clean_area 现在也与 evaluate 一致：sum([H,W,3])/(H*W)
                            _area_eval = np.sum(clean_mask) / (clean_mask.shape[0] * clean_mask.shape[1])
                    else:
                        print(f"[Mask] ⚠️ idx={idx}: cv2.imread 失败: {_used_mask_path}")

                elif hasattr(self, 'light_and_flare'):
                    # 最后备选（不可靠：可能是训练残留的同一张 light_and_flare）
                    _laf_mean = self.light_and_flare.mean().item()
                    print(f"[Mask] ⚠️ idx={idx}: 无磁盘 mask，使用 light_and_flare (mean={_laf_mean:.4f}，"
                          f"若所有图相同=训练残留！)")
                    seg_img = tensor2img([self.light_and_flare.detach().cpu()])
                    clean_mask = extract_mask_standard(seg_img)
                else:
                    print(f"[Mask] ⚠️ idx={idx}: 未找到任何 mask 来源，跳过该图！")

                # ── 计算指标 ────────────────────────────────────────────────
                if clean_mask is not None:
                    c_psnr = calculate_masked_psnr(sr_img, gt_img, clean_mask)
                    c_ssim = calculate_masked_ssim(sr_img, gt_img, clean_mask)
                    self.metric_results['clean_psnr'] += c_psnr
                    self.metric_results['clean_ssim'] += c_ssim

            del self.lq
            del self.output
            torch.cuda.empty_cache()

            img_name = 'deflare_' + str(idx).zfill(5) + '_'

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             img_name, f'{img_name}_{current_iter}.png')
                else:
                    suffix = self.opt['val']['suffix'] if self.opt['val']['suffix'] else self.opt['name']
                    save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                             f'{img_name}_{suffix}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                if not is_val_2:
                    for name, opt_ in self.opt['val']['metrics'].items():
                        self.metric_results[name] += calculate_metric(metric_data, opt_)

            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')

        if use_pbar:
            pbar.close()

        if with_metrics:
            keys_to_avg = ['clean_psnr', 'clean_ssim'] if is_val_2 else list(self.opt['val']['metrics'].keys())

            # ★ 用实际成功配对 mask 的图像数做分母，与 evaluate 的 clean_num 逻辑一致
            _divisor = _valid_mask_count if (is_val_2 and _valid_mask_count > 0) else (idx + 1)
            for metric in keys_to_avg:
                if metric in self.metric_results:
                    self.metric_results[metric] /= _divisor
                    self._update_best_metric_result(dataset_name, metric,
                                                    self.metric_results[metric], current_iter)

            self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def _log_validation_metric_values(self, current_iter, dataset_name, tb_logger):
        log_str = f'Validation {dataset_name}\n'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
            if hasattr(self, 'best_metric_results'):
                log_str += (f'\tBest: {self.best_metric_results[dataset_name][metric]["val"]:.4f} @ '
                            f'{self.best_metric_results[dataset_name][metric]["iter"]} iter')
            log_str += '\n'

        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{dataset_name}/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        if self.output_ch==3:
            self.blend= blend_light_source(self.lq, self.deflare, 0.97)
            out_dict['result']= self.blend.detach().cpu()
        elif self.output_ch ==6:
            out_dict['result']= self.deflare.detach().cpu()
        out_dict['flare']=self.flare_hat.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

