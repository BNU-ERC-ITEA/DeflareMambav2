import os
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
from glob import glob
import lpips
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import mean_squared_error as compare_mse
from skimage import io
from torchvision.transforms import ToTensor


def compare_lpips(img1, img2, loss_fn_alex):
    """计算两幅图像的LPIPS指标"""
    to_tensor = ToTensor()
    img1_tensor = to_tensor(img1).unsqueeze(0)
    img2_tensor = to_tensor(img2).unsqueeze(0)
    output_lpips = loss_fn_alex(img1_tensor.cuda(), img2_tensor.cuda())
    return output_lpips.cpu().detach().numpy()[0, 0, 0, 0]


def compare_score(img1, img2, img_seg, image_size=None):
    """计算PSNR-Clean和SSIM-Clean指标

    Args:
        image_size: 图像尺寸 (h, w)，如果为None则自动检测
    """
    mask_type_list = ['clean']
    metric_dict = {'clean': 0, 'ssim_clean': 0}

    for mask_type in mask_type_list:
        mask_area, img_mask = extract_mask(img_seg, image_size)[mask_type]
        if mask_area > 0:
            # ==== 计算 PSNR-Clean ====
            img_gt_masked = img1 * img_mask
            img_input_masked = img2 * img_mask
            input_mse = compare_mse(img_gt_masked, img_input_masked) / (255 * 255 * mask_area)
            input_psnr = 10 * np.log10((1.0 ** 2) / input_mse)
            metric_dict[mask_type] = input_psnr

            # ==== 计算 SSIM-Clean ====
            # 计算整张图的逐像素 SSIM Map
            _, ssim_map = compare_ssim(img1, img2, channel_axis=-1, full=True)
            # 利用二值化掩膜过滤，只求取掩膜覆盖区域内(原mask为黑色) SSIM 值的平均数
            ssim_clean_val = np.sum(ssim_map * img_mask) / np.sum(img_mask)
            metric_dict['ssim_clean'] = ssim_clean_val
        else:
            metric_dict.pop(mask_type, None)
            metric_dict.pop('ssim_clean', None)
    return metric_dict


def compare_score_new(img1, img2, img_seg, image_size=None):
    """计算G-PSNR, S-PSNR, GO-PSNR指标

    Args:
        image_size: 图像尺寸 (h, w)，如果为None则自动检测
    """
    mask_type_list = ['glare', 'streak', 'ghost']
    metric_dict = {'glare': 0, 'streak': 0, 'ghost': 0}

    for mask_type in mask_type_list:
        mask_area, img_mask = extract_mask_new(img_seg, image_size)[mask_type]
        if mask_area > 0:
            img_gt_masked = img1 * img_mask
            img_input_masked = img2 * img_mask
            input_mse = compare_mse(img_gt_masked, img_input_masked) / (255 * 255 * mask_area)
            input_psnr = 10 * np.log10((1.0 ** 2) / input_mse)
            metric_dict[mask_type] = input_psnr
        else:
            metric_dict.pop(mask_type)
    return metric_dict


def extract_mask_new(img_seg, image_size=None):
    """提取glare, streak, ghost掩码

    Args:
        img_seg: 分割图像
        image_size: 图像尺寸 (h, w)，如果为None则自动检测
    """
    mask_dict = {}

    # 获取图像尺寸
    if image_size:
        h, w = image_size
    else:
        h, w = img_seg.shape[:2]

    # 确保图像是三通道的
    if len(img_seg.shape) == 2:
        # 如果是灰度图，扩展为三通道
        img_seg = np.stack([img_seg, img_seg, img_seg], axis=2)

    # 处理掩码，确保值在0-255范围内
    if img_seg.max() > 1.0:
        # 如果值大于1，假设是0-255范围
        streak_mask = ((img_seg[:, :, 0].astype(float) - img_seg[:, :, 1].astype(float)) / 255.0).clip(0, 1)
        glare_mask = (img_seg[:, :, 1].astype(float) / 255.0).clip(0, 1)
        ghost_mask = (img_seg[:, :, 2].astype(float) / 255.0).clip(0, 1)
    else:
        # 如果值在0-1范围内
        streak_mask = ((img_seg[:, :, 0] - img_seg[:, :, 1])).clip(0, 1)
        glare_mask = img_seg[:, :, 1].clip(0, 1)
        ghost_mask = img_seg[:, :, 2].clip(0, 1)

    # 创建三通道掩码
    glare_mask_3ch = np.stack([glare_mask, glare_mask, glare_mask], axis=2)
    streak_mask_3ch = np.stack([streak_mask, streak_mask, streak_mask], axis=2)
    ghost_mask_3ch = np.stack([ghost_mask, ghost_mask, ghost_mask], axis=2)

    # 使用实际图像尺寸计算面积比例
    mask_dict['glare'] = [np.sum(glare_mask) / (h * w), glare_mask_3ch]
    mask_dict['streak'] = [np.sum(streak_mask) / (h * w), streak_mask_3ch]
    mask_dict['ghost'] = [np.sum(ghost_mask) / (h * w), ghost_mask_3ch]

    return mask_dict


def extract_mask(img_seg, image_size=None):
    """提取clean掩码

    注意：白色区域（值接近255）将被忽略，黑色区域（值接近0）将被考虑

    Args:
        img_seg: 分割图像
        image_size: 图像尺寸 (h, w)，如果为None则自动检测
    """
    mask_dict = {}

    # 获取图像尺寸
    if image_size:
        h, w = image_size
    else:
        h, w = img_seg.shape[:2]

    # 处理掩码，确保值在0-1范围内
    if img_seg.max() > 1.0:
        # 反转掩码：白色(255)变成0，黑色(0)变成1
        clean_mask = 1.0 - (img_seg.astype(float) / 255.0).clip(0, 1)
    else:
        # 反转掩码：白色(1.0)变成0，黑色(0.0)变成1
        clean_mask = 1.0 - img_seg.astype(float).clip(0, 1)

    # 确保所有值在0-1范围内
    clean_mask = clean_mask.clip(0, 1)

    # 如果是单通道掩码，扩展为三通道
    if len(clean_mask.shape) == 2:
        clean_mask_3ch = np.stack([clean_mask, clean_mask, clean_mask], axis=2)
    elif len(clean_mask.shape) == 3 and clean_mask.shape[2] == 3:
        clean_mask_3ch = clean_mask
    else:
        # 如果通道数不是3，只取第一个通道并复制
        clean_mask_3ch = np.stack([clean_mask[:, :, 0], clean_mask[:, :, 0], clean_mask[:, :, 0]], axis=2)

    # 计算掩码面积（黑色区域的总和）
    mask_dict['clean'] = [np.sum(clean_mask) / (h * w), clean_mask_3ch]

    return mask_dict


def check_image_sizes(pred_list, gt_list, mask_list=None):
    """检查图像尺寸并返回一致的尺寸信息"""
    sizes = {}

    # 检查预测图像尺寸
    pred_sizes = []
    for pred_path in pred_list[:3]:  # 只检查前3张
        img = io.imread(pred_path)
        pred_sizes.append(img.shape[:2])

    # 检查真实图像尺寸
    gt_sizes = []
    for gt_path in gt_list[:3]:  # 只检查前3张
        img = io.imread(gt_path)
        gt_sizes.append(img.shape[:2])

    # 检查掩码尺寸（如果存在）
    mask_sizes = []
    if mask_list:
        for mask_path in mask_list[:3]:  # 只检查前3张
            img = io.imread(mask_path)
            mask_sizes.append(img.shape[:2])

    # 确定主要尺寸
    def get_most_common_size(sizes):
        if not sizes:
            return None
        unique_sizes, counts = np.unique(sizes, axis=0, return_counts=True)
        most_common_idx = np.argmax(counts)
        return tuple(unique_sizes[most_common_idx])

    pred_size = get_most_common_size(pred_sizes)
    gt_size = get_most_common_size(gt_sizes)
    mask_size = get_most_common_size(mask_sizes) if mask_sizes else None

    # 检查尺寸是否一致
    all_sizes = [s for s in [pred_size, gt_size, mask_size] if s is not None]
    consistent = all(s == all_sizes[0] for s in all_sizes)

    # 如果不一致，使用真实图像尺寸作为参考
    if not consistent:
        print("警告: 图像尺寸不一致")
        print(f"预测图像尺寸: {pred_sizes}")
        print(f"真实图像尺寸: {gt_sizes}")
        if mask_sizes:
            print(f"掩码图像尺寸: {mask_sizes}")

        # 使用真实图像尺寸作为目标尺寸
        target_size = gt_size
        print(f"将使用真实图像尺寸作为目标: {target_size}")
    else:
        target_size = all_sizes[0]

    # 存储尺寸信息
    sizes['pred_size'] = pred_size
    sizes['gt_size'] = gt_size
    sizes['mask_size'] = mask_size
    sizes['target_size'] = target_size
    sizes['consistent'] = consistent

    return sizes


def resize_to_target(img, target_size):
    """将图像调整到目标尺寸"""
    from skimage.transform import resize
    if img.shape[:2] != target_size:
        return resize(img, target_size, preserve_range=True).astype(img.dtype)
    return img


def calculate_metrics(pred_folder, gt_folder, mask_folder=None, metric_type='standard', force_resize=False):
    """
    计算预测图像与真实图像之间的评估指标

    Args:
        pred_folder: 预测图像文件夹
        gt_folder: 真实图像文件夹
        mask_folder: 掩码图像文件夹 (可选)
        metric_type: 指标类型 ('standard' 或 'new')
        force_resize: 是否强制调整所有图像到相同尺寸

    Returns:
        dict: 包含所有评估指标的字典
    """
    # 初始化LPIPS
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()

    # 获取图像列表
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']

    gt_list = []
    for ext in image_extensions:
        gt_list.extend(glob(os.path.join(gt_folder, ext)))
    gt_list = sorted(gt_list)

    pred_list = []
    for ext in image_extensions:
        pred_list.extend(glob(os.path.join(pred_folder, ext)))
    pred_list = sorted(pred_list)

    if mask_folder:
        mask_list = []
        for ext in image_extensions:
            mask_list.extend(glob(os.path.join(mask_folder, ext)))
        mask_list = sorted(mask_list)

    # 检查图像数量
    if len(gt_list) == 0:
        raise ValueError(f"在 {gt_folder} 中没有找到真实图像")
    if len(pred_list) == 0:
        raise ValueError(f"在 {pred_folder} 中没有找到预测图像")

    print(f"找到 {len(gt_list)} 张真实图像和 {len(pred_list)} 张预测图像")

    # 检查图像尺寸
    sizes = check_image_sizes(pred_list, gt_list, mask_list if mask_folder else None)
    target_size = sizes['target_size']

    print(f"检测到图像尺寸:")
    print(f"  预测图像: {sizes['pred_size']}")
    print(f"  真实图像: {sizes['gt_size']}")
    if mask_folder:
        print(f"  掩码图像: {sizes['mask_size']}")
    print(f"  目标尺寸: {target_size}")

    if not sizes['consistent'] and not force_resize:
        print("\n警告: 图像尺寸不一致！这可能影响评估结果。")
        print("可以使用 --force_resize 参数强制调整所有图像到相同尺寸。")
        response = input("是否继续? (y/n): ")
        if response.lower() != 'y':
            print("评估已取消。")
            exit(0)

    # 如果数量不匹配，使用较小的数量
    if len(gt_list) != len(pred_list):
        print(f"警告: 真实图像数量 ({len(gt_list)}) 与预测图像数量 ({len(pred_list)}) 不匹配")
        min_len = min(len(gt_list), len(pred_list))
        gt_list = gt_list[:min_len]
        pred_list = pred_list[:min_len]
        print(f"将评估前 {min_len} 张图像")

    if mask_folder and mask_list:
        # 确保掩码数量匹配
        if len(mask_list) != len(gt_list):
            print(f"警告: 掩码图像数量 ({len(mask_list)}) 与真实图像数量 ({len(gt_list)}) 不匹配")
            min_len = min(len(gt_list), len(mask_list))
            gt_list = gt_list[:min_len]
            pred_list = pred_list[:min_len]
            mask_list = mask_list[:min_len]

    n = len(gt_list)

    # 初始化指标
    ssim, psnr, lpips_val = 0, 0, 0

    if metric_type == 'standard':
        score_dict = {'clean': 0, 'clean_num': 0, 'ssim_clean': 0, 'ssim_clean_num': 0}
        result_file_name = 'evaluation_results.txt'
    else:
        score_dict = {'glare': 0, 'streak': 0, 'ghost': 0, 'glare_num': 0, 'streak_num': 0, 'ghost_num': 0}
        result_file_name = 'evaluation_results_new.txt'

    # 创建结果文件
    result_dir = os.path.dirname(pred_folder) if os.path.isdir(pred_folder) else pred_folder
    file_path = os.path.join(result_dir, result_file_name)
    file = open(file_path, 'w')

    print(f"开始评估 {n} 对图像...")

    for i in tqdm(range(n)):
        try:
            # 加载图像
            img_gt = io.imread(gt_list[i])
            img_pred = io.imread(pred_list[i])

            # 如果需要，调整图像尺寸
            if force_resize or not sizes['consistent']:
                img_gt = resize_to_target(img_gt, target_size)
                img_pred = resize_to_target(img_pred, target_size)

            # 验证尺寸匹配
            if img_gt.shape[:2] != img_pred.shape[:2]:
                print(f"警告: 图像 {i} 尺寸不匹配: GT={img_gt.shape[:2]}, Pred={img_pred.shape[:2]}")
                from skimage.transform import resize
                img_pred = resize(img_pred, img_gt.shape[:2], preserve_range=True).astype(img_pred.dtype)

            # 计算基础指标
            ssim0 = compare_ssim(img_gt, img_pred, channel_axis=-1)
            psnr0 = compare_psnr(img_gt, img_pred, data_range=255)
            lpips_val0 = compare_lpips(img_gt, img_pred, loss_fn_alex)

            ssim += ssim0
            psnr += psnr0
            lpips_val += lpips_val0

            # 如果有掩码，计算额外指标
            extra_metric = 0
            extra_metric_ssim = 0
            if mask_folder and mask_list and i < len(mask_list):
                img_seg = io.imread(mask_list[i])

                # 如果需要，调整掩码尺寸
                if force_resize or not sizes['consistent']:
                    img_seg = resize_to_target(img_seg, target_size)

                if metric_type == 'standard':
                    metric_dict = compare_score(img_gt, img_pred, img_seg, target_size)
                    for key in metric_dict.keys():
                        if key not in score_dict:
                            score_dict[key] = 0
                            score_dict[key + '_num'] = 0
                        score_dict[key] += metric_dict[key]
                        score_dict[key + '_num'] += 1
                    extra_metric = metric_dict.get('clean', 0)
                    extra_metric_ssim = metric_dict.get('ssim_clean', 0)
                else:
                    metric_dict = compare_score_new(img_gt, img_pred, img_seg, target_size)
                    for key in metric_dict.keys():
                        score_dict[key] += metric_dict[key]
                        score_dict[key + '_num'] += 1
                    extra_metric = metric_dict.get('glare', 0)  # 示例，使用glare作为额外指标

            # 写入每个图像的结果
            if metric_type == 'standard':
                if mask_folder and mask_list:
                    text_content = f"{os.path.basename(pred_list[i])}, PSNR: {psnr0:.4f}, SSIM: {ssim0:.4f}, LPIPS: {lpips_val0:.4f}, PSNR-Clean: {extra_metric:.4f}, SSIM-Clean: {extra_metric_ssim:.4f}\n"
                else:
                    text_content = f"{os.path.basename(pred_list[i])}, PSNR: {psnr0:.4f}, SSIM: {ssim0:.4f}, LPIPS: {lpips_val0:.4f}\n"
            else:
                text_content = f"{os.path.basename(pred_list[i])}, PSNR: {psnr0:.4f}, SSIM: {ssim0:.4f}, LPIPS: {lpips_val0:.4f}\n"

            file.write(text_content)

        except Exception as e:
            print(f"处理图像 {pred_list[i]} 时出错: {e}")
            import traceback
            traceback.print_exc()

    # 计算平均值
    ssim /= n
    psnr /= n
    lpips_val /= n

    # 写入平均结果
    text_content = f"\n平均结果 (基于 {n} 张图像):\n"
    text_content += f"图像尺寸: {target_size[1]}x{target_size[0]}\n"
    text_content += f"PSNR: {psnr:.4f}\n"
    text_content += f"SSIM: {ssim:.4f}\n"
    text_content += f"LPIPS: {lpips_val:.4f}\n"

    if metric_type == 'standard' and mask_folder and score_dict['clean_num'] > 0:
        score_dict['clean'] /= score_dict['clean_num']
        text_content += f"PSNR-Clean: {score_dict['clean']:.4f}\n"
        if score_dict.get('ssim_clean_num', 0) > 0:
            score_dict['ssim_clean'] /= score_dict['ssim_clean_num']
            text_content += f"SSIM-Clean: {score_dict['ssim_clean']:.4f}\n"

    elif metric_type == 'new' and mask_folder and mask_list:
        # 计算新的评分标准
        for key in ['glare', 'streak', 'ghost']:
            if score_dict[key + '_num'] > 0:
                score_dict[key] /= score_dict[key + '_num']

        if all(score_dict[key + '_num'] > 0 for key in ['glare', 'streak', 'ghost']):
            score_dict['score'] = 1 / 3 * (score_dict['glare'] + score_dict['ghost'] + score_dict['streak'])
            text_content += f"Score: {score_dict['score']:.4f}\n"
            text_content += f"G-PSNR: {score_dict['glare']:.4f}\n"
            text_content += f"S-PSNR: {score_dict['streak']:.4f}\n"
            text_content += f"GO-PSNR: {score_dict['ghost']:.4f}\n"

    print(text_content)
    file.write(text_content)
    file.close()

    print(f"详细结果已保存到: {file_path}")

    # 返回结果字典
    results = {
        'image_size': target_size,
        'num_images': n,
        'PSNR': psnr,
        'SSIM': ssim,
        'LPIPS': lpips_val,
    }

    if metric_type == 'standard' and mask_folder and score_dict['clean_num'] > 0:
        results['PSNR-Clean'] = score_dict['clean']
        if score_dict.get('ssim_clean_num', 0) > 0:
            results['SSIM-Clean'] = score_dict['ssim_clean']
    elif metric_type == 'new' and mask_folder and mask_list:
        results['Score'] = score_dict.get('score', None)
        results['G-PSNR'] = score_dict.get('glare', None)
        results['S-PSNR'] = score_dict.get('streak', None)
        results['GO-PSNR'] = score_dict.get('ghost', None)

    return results


def main():
    parser = argparse.ArgumentParser(description='兼容版本评估脚本 - 支持512x512和3024x3024图像，计算包含SSIM-Clean')
    parser.add_argument('--pred', type=str,
                        default="result/FlareX/deflare",
                        help='预测图像文件夹路径')
    parser.add_argument('--gt', type=str,
                        default="testdata/FlareX/gt",
                        help='真实图像文件夹路径')
    parser.add_argument('--mask', type=str,
                        default="testdata/FlareX/mask",
                        help='掩码图像文件夹路径 (可选)')
    parser.add_argument('--metric_type', type=str, default='standard',
                        choices=['standard', 'new'],
                        help='指标类型: standard (PSNR, SSIM, LPIPS, PSNR-Clean, SSIM-Clean) 或 new (包含Score, G-PSNR等)')
    parser.add_argument('--force_resize', action='store_true',
                        help='强制调整所有图像到相同尺寸')

    args = parser.parse_args()

    # 验证输入路径
    if not os.path.exists(args.pred):
        print(f"错误: 预测图像文件夹 '{args.pred}' 不存在")
        exit(1)

    if not os.path.exists(args.gt):
        print(f"错误: 真实图像文件夹 '{args.gt}' 不存在")
        exit(1)

    if args.mask and not os.path.exists(args.mask):
        print(f"错误: 掩码图像文件夹 '{args.mask}' 不存在")
        exit(1)

    print(f"预测图像来自: {args.pred}")
    print(f"真实图像来自: {args.gt}")
    if args.mask:
        print(f"掩码图像来自: {args.mask}")
    print(f"指标类型: {args.metric_type}")
    if args.force_resize:
        print("启用强制调整尺寸模式")

    try:
        results = calculate_metrics(args.pred, args.gt, args.mask, args.metric_type, args.force_resize)
        print("\n评估任务成功完成!")
    except Exception as e:
        print(f"评估过程中出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()