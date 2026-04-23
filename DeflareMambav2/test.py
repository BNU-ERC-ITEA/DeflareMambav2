import glob
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageChops

import argparse
from basicsr.archs.DeflareMambav2_changed_arch import DeflareMambav2
from basicsr.utils.flare_util import blend_light_source, mkdir, \
    predict_flare_from_6_channel, predict_flare_from_3_channel
import torchvision.transforms as transforms
import os

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='testdata/FlareX/input')
parser.add_argument('--output', type=str,
                    default='result/FlareX')
# parser.add_argument('--input', type=str, default='testdata/Flare7k-real/input')
# parser.add_argument('--output', type=str,
#                     default='result/Flare7k-real')
parser.add_argument('--model_type', type=str, default='DeflareMambav2')
parser.add_argument('--model_path', type=str,
                    default='weight/main.pth')
parser.add_argument('--lpg_model_path', type=str,
                    default='weight/FPN.pth',
                    help='Path to FPN model checkpoint')
parser.add_argument('--output_ch', type=int, default=6)
parser.add_argument('--flare7kpp', action='store_const', const=True,
                    default=False)  # use flare7kpp's inference method and output the light source directly.

args = parser.parse_args()
model_type = args.model_type
images_path = os.path.join(args.input, "*.*")
result_path = args.output
pretrain_dir = args.model_path
lpg_model_path = args.lpg_model_path
output_ch = args.output_ch


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def load_params(model_path):
    #  full_model=torch.load(model_path)
    full_model = torch.load(model_path, map_location=torch.device('cpu'))
    if 'params_ema' in full_model:
        return full_model['params_ema']
    elif 'params' in full_model:
        return full_model['params']
    else:
        return full_model


class ImageProcessor:
    def __init__(self, model, lpg_model=None):
        self.model = model
        self.lpg_model = lpg_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def resize_image(self, image, target_size):
        original_width, original_height = image.size
        aspect_ratio = original_width / original_height

        if original_width < original_height:
            new_width = target_size
            new_height = int(target_size / aspect_ratio)
        else:
            new_height = target_size
            new_width = int(target_size * aspect_ratio)

        return image.resize((new_width, new_height))

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
    def process_image(self, image):
        # Open the original image
        to_tensor = transforms.ToTensor()
        original_image = image

        # Resize the image proportionally to make the shorter side 512 pixels
        resized_image = self.resize_image(original_image, 512)
        resized_width, resized_height = resized_image.size

        # Process each 512-pixel segment separately
        segments = []
        overlaps = []
        if resized_width > 512:
            for end_x in range(512, resized_width + 256, 256):
                end_x = min(end_x, resized_width)
                overlaps.append(end_x)
                cropped_image = resized_image.crop((end_x - 512, 0, end_x, 512))

                segment_tensor = to_tensor(cropped_image).unsqueeze(0).to(self.device)

                if self.lpg_model is not None:
                    with torch.no_grad():
                        # LPG模型输出应该是两个张量
                        # first_output, lpg_output = self.lpg_model(segment_tensor)
                        first_output, lpg_output, logits, pred_positions, pred_conf = self.lpg_model(segment_tensor)

                        K_limit = self.max_sources_main_train
                        pred_positions = pred_positions[:, :K_limit, :]  # [B, 1, 2]
                        pred_conf = pred_conf[:, :K_limit]  # [B, 1]

                        H, W = self.lq.shape[2], self.lq.shape[3]
                        import torch.nn.functional as F

                        # ── ② heatmap 上采样回原始分辨率 ─────────────────────────
                        lpg_output_full = F.interpolate(
                            lpg_output,
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
                        # 确保输出形状正确
                        first_output = first_output.squeeze(0)
                        lpg_output_full = lpg_output.squeeze(0)
                        pred_positions = pred_positions.squeeze(0)

                    # 拼接输入和LPG输出: 3 + 1 + 1 +1 = 6通道
                    concat_input = torch.cat((segment_tensor.squeeze(0), first_output, lpg_output_full,pred_positions), dim=0).unsqueeze(0)

                    processed_segment = self.model(concat_input).squeeze(0)
                else:
                    processed_segment = self.model(segment_tensor).squeeze(0)

                segments.append(processed_segment)
        else:
            for end_y in range(512, resized_height + 256, 256):
                end_y = min(end_y, resized_height)
                overlaps.append(end_y)
                cropped_image = resized_image.crop((0, end_y - 512, 512, end_y))

                segment_tensor = to_tensor(cropped_image).unsqueeze(0).to(self.device)

                if self.lpg_model is not None:
                    with torch.no_grad():
                        first_output, lpg_output, logits, pred_positions, pred_conf = self.lpg_model(segment_tensor)

                        K_limit = 6
                        pred_positions = pred_positions[:, :K_limit, :]  # [B, 1, 2] -> 实际应该是 [B, K, 2]
                        pred_conf = pred_conf[:, :K_limit]  # [B, 1] -> 实际应该是 [B, K]

                        H, W = 512, 512
                        import torch.nn.functional as F

                        # ── ② heatmap 上采样回原始分辨率 ─────────────────────────
                        lpg_output_full = F.interpolate(
                            lpg_output,
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

                        # 确保输出形状正确 (去除 Batch 维度)
                        first_output = first_output.squeeze(0)
                        # 【修复1】这里必须是 lpg_output_full，不能是 lpg_output
                        lpg_output_full = lpg_output_full.squeeze(0)
                        # 【修复2】处理 pos_map 以备拼接
                        pos_map = pos_map.squeeze(0)

                        # 拼接输入和LPG输出: 3 + 1 + 1 + 1 = 6通道
                        # 【修复3】将 pred_positions 替换为 pos_map
                    concat_input = torch.cat((segment_tensor.squeeze(0), first_output, lpg_output_full, pos_map),
                                             dim=0).unsqueeze(0)
                    processed_segment = self.model(concat_input).squeeze(0)
                else:
                    processed_segment = self.model(segment_tensor).squeeze(0)

                segments.append(processed_segment)
        overlaps = [0] + [prev - cur + 512 for prev, cur in zip(overlaps[:-1], overlaps[1:])]

        # Blending the segments
        for i in range(1, len(segments)):
            overlap = overlaps[i]
            alpha = torch.linspace(0, 1, steps=overlap).to(self.device)
            if resized_width > 512:
                alpha = alpha.view(1, -1, 1).expand(512, -1, 6).permute(2, 0, 1)
                segments[i][:, :, :overlap] = alpha * segments[i][:, :, :overlap] + (1 - alpha) * segments[i - 1][:, :,
                                                                                                  -overlap:]
            else:
                alpha = alpha.view(-1, 1, 1).expand(-1, 512, 6).permute(2, 0, 1)
                segments[i][:, :overlap, :] = alpha * segments[i][:, :overlap, :] + (1 - alpha) * segments[i - 1][:,
                                                                                                  -overlap:, :]

        # Concatenating all the segments
        if resized_width > 512:
            blended = [segment[:, :, :-overlap] for segment, overlap in zip(segments[:-1], overlaps[1:])] + [
                segments[-1]]
            merged_image = torch.cat(blended, dim=2)
        else:
            blended = [segment[:, :-overlap, :] for segment, overlap in zip(segments[:-1], overlaps[1:])] + [
                segments[-1]]
            merged_image = torch.cat(blended, dim=1)

        return merged_image


def demo(images_path, output_path, model_type, output_ch, pretrain_dir, flare7kpp_flag, lpg_model_path=None):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    test_path = sorted(glob.glob(images_path))
    result_path = output_path
    os.makedirs(result_path, exist_ok=True)
    torch.cuda.empty_cache()

    lpg_model = None
    in_chans = 3  # 默认输入通道数

    print("=" * 80)
    print("模型加载调试信息")
    print("=" * 80)

    if model_type == 'Uformer':
        print(f"创建Uformer模型...")
        model = Uformer(img_size=512, img_ch=3, output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
        print("Uformer模型加载完成")

    elif model_type == 'U_Net' or model_type == 'U-Net':
        print(f"创建U_Net模型...")
        model = U_Net(img_ch=3, output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
        print("U_Net模型加载完成")

    elif model_type == 'DeflareMamba':
        print(f"创建DeflareMamba模型...")
        model = DeflareMamba(img_size=512, in_chans=3, output_ch=6, img_range=1., d_state=10,
                             depths=[1, 2, 4, 4, 4, 2, 1], embed_dim=40, mlp_ratio=1.).cuda()
        model.load_state_dict(load_params(pretrain_dir))
        print("DeflareMamba模型加载完成")


    elif model_type == 'FGRNet':
        print(f"创建FGRNet模型...")
        model = FGRNet(img_size=512, img_ch=3, output_ch=output_ch).cuda()
        model.load_state_dict(load_params(pretrain_dir))
        print("创建FGRNet模型")

    elif model_type == 'DeflareMambav2':
        print("\n=== 创建 DeflareMambav2 模型 ===")

        # 1. 加载LPG模型（如果指定）
        if lpg_model_path and os.path.exists(lpg_model_path):
            print(f"尝试加载LPG模型: {lpg_model_path}")
            try:
                # from basicsr.models.lpg import LPG_FrequencyLite
                # lpg_model = LPG_FrequencyLite().cuda()
                from basicsr.models.FPN import FPN
                lpg_model = FPN().cuda()
                # 先加载LPG权重
                lpg_state_dict = load_params(lpg_model_path)

                # 检查是否需要从'model_state_dict'中提取
                if 'model_state_dict' in lpg_state_dict:
                    print("从'model_state_dict'中提取LPG权重")
                    lpg_state_dict = lpg_state_dict['model_state_dict']

                lpg_model.load_state_dict(lpg_state_dict)
                lpg_model.eval()
                print("✓ LPG模型加载成功")

                # 测试LPG模型输出
                with torch.no_grad():
                    test_input = torch.randn(1, 3, 512, 512).cuda()
                    # first_output, lpg_output = lpg_model(test_input)
                    flare_map, heatmap, logits, pred_positions, pred_conf = lpg_model(test_input)
                    print(f"  LPG测试输出形状: first_output={flare_map.shape}, lpg_output={heatmap.shape}")


                in_chans = 3


            except Exception as e:
                print(f"✗ 加载LPG模型失败: {e}")
                lpg_model = None
                in_chans = 3
        else:
            print("未提供LPG模型路径或文件不存在，不使用LPG")
            in_chans = 3

        # 2. 创建主模型

        # 创建真正的模型
        print(f"创建真正的模型 (GPU)...")
        model = DeflareMambav2(
            img_size=512,
            in_chans=in_chans,
            output_ch=6,
            img_range=1.,
            d_state=10,
            depths=[2, 2, 2, 2, 2, 2, 2],
            num_heads=[2, 2, 2, 2, 2, 2, 2],
            embed_dim=36,
            mlp_ratio=1.
        ).cuda()

        # 3. 加载权重
        print(f"\n加载权重: {pretrain_dir}")
        state_dict = load_params(pretrain_dir)
        print(f"权重文件中的参数数量: {len(state_dict)}")

        # 5. 检查权重和模型的匹配情况
        print(f"\n检查权重和模型的匹配情况:")

        # 模型参数形状
        model_params = {name: param.shape for name, param in model.named_parameters()}

        # 检查哪些参数不匹配
        mismatch_count = 0
        for key, weight_value in state_dict.items():
            if key in model_params:
                model_shape = model_params[key]
                if isinstance(weight_value, torch.Tensor):
                    weight_shape = weight_value.shape
                    if weight_shape != model_shape:
                        mismatch_count += 1
                        print(f"  ⚠️ 不匹配: {key}")
                        print(f"    模型形状: {model_shape}, 权重形状: {weight_shape}")
                else:
                    print(f"  ⚠️ 权重不是张量: {key} (类型: {type(weight_value)})")
            else:
                print(f"  ⚠️ 缺失: {key} (在模型中不存在)")

        print(f"总不匹配参数数量: {mismatch_count}")

        # 6. 处理不匹配的参数
        print(f"\n处理不匹配的参数...")

        # 过滤掉不匹配的relative_position_bias_table参数
        keys_to_remove = []
        for key in list(state_dict.keys()):
            if 'relative_position_bias_table' in key:
                # 检查形状是否匹配
                if key in model_params:
                    weight_value = state_dict[key]
                    if isinstance(weight_value, torch.Tensor):
                        if weight_value.shape != model_params[key]:
                            print(f"  移除不匹配参数: {key}")
                            print(f"    权重形状: {weight_value.shape}, 模型形状: {model_params[key]}")
                            keys_to_remove.append(key)
                    else:
                        print(f"  移除非张量参数: {key}")
                        keys_to_remove.append(key)
                else:
                    print(f"  移除不存在的参数: {key}")
                    keys_to_remove.append(key)

        for key in keys_to_remove:
            del state_dict[key]

        print(f"移除了 {len(keys_to_remove)} 个不匹配的参数")

        # 7. 加载权重
        print(f"\n加载权重到模型...")
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

        if len(missing_keys) > 0:
            print(f"\n缺失的键 ({len(missing_keys)}个):")
            for i, key in enumerate(missing_keys[:20]):  # 只显示前20个
                print(f"  {i + 1}. {key}")
            if len(missing_keys) > 20:
                print(f"  ... (还有{len(missing_keys) - 20}个)")

        if len(unexpected_keys) > 0:
            print(f"\n意外的键 ({len(unexpected_keys)}个):")
            for i, key in enumerate(unexpected_keys[:20]):  # 只显示前20个
                print(f"  {i + 1}. {key}")
            if len(unexpected_keys) > 20:
                print(f"  ... (还有{len(unexpected_keys) - 20}个)")

        # 8. 最终检查
        print(f"\n最终模型状态检查:")
        print(f"  模型设备: {next(model.parameters()).device}")
        print(f"  模型是否在训练模式: {model.training}")

        if lpg_model:
            print(f"  LPG模型设备: {next(lpg_model.parameters()).device}")
            print(f"  LPG模型是否在训练模式: {lpg_model.training}")

    else:
        assert False, "This model is not supported!!"

    print("\n" + "=" * 80)
    print("开始处理图像")
    print("=" * 80)

    processor = ImageProcessor(model, lpg_model)
    to_tensor = transforms.ToTensor()

    for i, image_path in tqdm(enumerate(test_path)):
        img_name = os.path.basename(image_path)
        print(f"\n处理图像 {i + 1}/{len(test_path)}: {img_name}")

        if not flare7kpp_flag:
            mkdir(os.path.join(result_path, "deflare/"))
            deflare_path = os.path.join(result_path, "deflare/", img_name)

        mkdir(os.path.join(result_path, "flare/"))
        mkdir(os.path.join(result_path, "blend/"))

        flare_path = os.path.join(result_path, "flare/", img_name)
        blend_path = os.path.join(result_path, "blend/", img_name)

        merge_img = Image.open(image_path).convert("RGB")

        model.eval()
        if lpg_model:
            lpg_model.eval()

        with torch.no_grad():
            # 处理图像
            output_img = processor.process_image(merge_img).unsqueeze(0)
            print(f"  模型输出形状: {output_img.shape}")

            gamma = torch.Tensor([2.2])
            if output_ch == 6:
                deflare_img, flare_img_predicted, merge_img_predicted = predict_flare_from_6_channel(output_img, gamma)

                print(f"  deflare_img形状: {deflare_img.shape}")
                print(f"  flare_img_predicted形状: {flare_img_predicted.shape}")

                # 检查各通道的统计信息
                if deflare_img.shape[1] >= 3:
                    r_mean = deflare_img[0, 0, :, :].mean().item()
                    g_mean = deflare_img[0, 1, :, :].mean().item()
                    b_mean = deflare_img[0, 2, :, :].mean().item()

                    # 检查是否接近灰度图
                    diff_rg = abs(r_mean - g_mean)
                    diff_rb = abs(r_mean - b_mean)
                    diff_gb = abs(g_mean - b_mean)
                    if diff_rg < 0.01 and diff_rb < 0.01:
                        print(f"  ⚠️ 警告: 输出接近灰度图 (通道差异很小)")

            elif output_ch == 3:
                flare_mask = torch.zeros_like(merge_img)
                deflare_img, flare_img_predicted = predict_flare_from_3_channel(output_img, flare_mask, output_img,
                                                                                merge_img, merge_img, gamma)
            else:
                assert False, "This output_ch is not supported!!"

            if not flare7kpp_flag:
                print(f"  保存deflare图像到: {deflare_path}")
                torchvision.utils.save_image(deflare_img, deflare_path)

                blend_input = to_tensor(processor.resize_image(merge_img, 512)).cuda().unsqueeze(0)
                deflare_img = blend_light_source(blend_input, deflare_img, 0.95)

            # 转换为PIL图像
            deflare_img_np = deflare_img.squeeze(0).permute(1, 2, 0).clamp(0, 1).cpu().numpy()

            print(f"  numpy数组形状: {deflare_img_np.shape}")

            deflare_img_np_uint8 = (deflare_img_np * 255).astype(np.uint8)

            deflare_img_pil = Image.fromarray(deflare_img_np_uint8, 'RGB')
            print(f"  PIL图像模式: {deflare_img_pil.mode}")

            # 计算炫光图像和混合图像
            flare_img_orig = ImageChops.difference(merge_img.resize(deflare_img_pil.size), deflare_img_pil)
            deflare_img_orig = ImageChops.difference(merge_img,
                                                     flare_img_orig.resize(merge_img.size, resample=Image.BICUBIC))

            # 保存结果
            flare_img_orig.save(flare_path)
            deflare_img_orig.save(blend_path)

            print(f"  保存完成:")
            print(f"    flare图像: {flare_path}")
            print(f"    blend图像: {blend_path}")

        print(f"  图像 {img_name} 处理完成")
        print("-" * 60)

    print("\n" + "=" * 80)
    print("所有图像处理完成")
    print("=" * 80)


demo(images_path, result_path, model_type, output_ch, pretrain_dir, args.flare7kpp, args.lpg_model_path)