import torch
from collections import OrderedDict
from os import path as osp
from tqdm import tqdm

from basicsr.archs import build_network
from basicsr.losses import build_loss
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.registry import MODEL_REGISTRY
from .base_model import BaseModel


@MODEL_REGISTRY.register()
class SRModel(BaseModel):
    """Base SR model for single image super-resolution."""

    def __init__(self, opt):
        super(SRModel, self).__init__(opt)

        # define network
        self.net_g = build_network(opt['network_g'])
        self.net_g = self.model_to_device(self.net_g)
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            param_key = self.opt['path'].get('param_key_g', 'params')
            self.load_network(self.net_g, load_path, self.opt['path'].get('strict_load_g', True), param_key)

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

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
        if train_opt.get('pixel_opt'):
            self.cri_pix = build_loss(train_opt['pixel_opt']).to(self.device)
        else:
            self.cri_pix = None

        if train_opt.get('perceptual_opt'):
            self.cri_perceptual = build_loss(train_opt['perceptual_opt']).to(self.device)
        else:
            self.cri_perceptual = None

        if self.cri_pix is None and self.cri_perceptual is None:
            raise ValueError('Both pixel and perceptual losses are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        # ⭐ 验证参数是否重叠
        if hasattr(self, 'lpg') :
            # 计算参数数量（考虑参数共享）
            g_params = sum(p.numel() for p in self.net_g.parameters())
            lpg_params = sum(p.numel() for p in self.lpg.parameters())

            # 计算共享参数数量
            g_param_ids = {id(p): p.numel() for p in self.net_g.parameters()}
            lpg_param_ids = {id(p): p.numel() for p in self.lpg.parameters()}
            overlap = sum(g_param_ids[pid] for pid in g_param_ids if pid in lpg_param_ids)

            print(f"net_g 参数量: {g_params / 1e6:.2f}M")
            print(f"lpg 参数量: {lpg_params / 1e6:.2f}M")
            print(f"重叠参数量: {overlap / 1e6:.2f}M")

            # 计算去重后的总参数量
            total_params = g_params + lpg_params - overlap
            print(f"去重后总参数量: {total_params / 1e6:.2f}M")

            if overlap:
                print("⚠️ 警告：optimizer_g 和 optimizer_lpg 管理了相同的参数！")
        try:
            if 'optim_g' not in train_opt:
                raise KeyError("Optimizer configuration 'optim_g' not found in training options.")

            g_params = sum(p.numel() for p in self.net_g.parameters())
            print(f"net_g 参数量: {g_params / 1e6:.2f}M")

            # 打印优化器选项
            optim_config = train_opt['optim_g']
            print("Current optimizer options:", optim_config)

            # 确保 type 字段存在
            optimizer_type = optim_config.get('type')
            if optimizer_type is None:
                raise ValueError("Optimizer type is missing in configuration.")

            # 输出当前学习率和权重衰减设置
            lr = optim_config['lr']
            weight_decay = optim_config.get('weight_decay', 0)
            print(f"Using {optimizer_type} optimizer with lr={lr} and weight_decay={weight_decay}.")

            # 使用对应的优化器类型
            if optimizer_type == 'Adam':
                self.optimizer_g = torch.optim.Adam(self.net_g.parameters(), lr=lr, weight_decay=weight_decay)
            else:
                raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

            # 确保优化器成功创建后，添加到优化器列表中
            self.optimizers.append(self.optimizer_g)
            print("Optimizer successfully created:", self.optimizer_g)

            # >>>>>>> 新增：LPG 优化器（与原代码风格保持一致） >>>>>>>
            if train_opt.get('lpg', {}).get('train_lpg', False):          # 1. 开关
                # 2. 优先用独立配置，否则复用主网络配置
                lpg_cfg = train_opt.get('optim_lpg', optim_config)
                print("LPG optimizer options:", lpg_cfg)

                lpg_type = lpg_cfg.get('type')
                if lpg_type is None:
                    raise ValueError("LPG optimizer type is missing in configuration.")

                lpg_lr = lpg_cfg['lr']
                lpg_wd = lpg_cfg.get('weight_decay', 0)
                print(f"Using LPG {lpg_type} optimizer with lr={lpg_lr} and weight_decay={lpg_wd}.")

                if lpg_type == 'Adam':
                    self.optimizer_lpg = torch.optim.Adam(self.lpg.parameters(), lr=lpg_lr, weight_decay=lpg_wd)

                else:
                    raise ValueError(f"Unsupported LPG optimizer type: {lpg_type}")

                self.optimizers.append(self.optimizer_lpg)
                print("LPG optimizer successfully created:", self.optimizer_lpg)
            # <<<<<<< 新增结束 <<<<<<<

            # 检查优化器列表长度
            print(f"Total number of optimizers after setup: {len(self.optimizers)}")

        except Exception as e:
            print("An error occurred in setup_optimizers:", str(e))



    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt)
            l_total += l_pix
            loss_dict['l_pix'] = l_pix
        # perceptual loss
        if self.cri_perceptual:
            l_percep, l_style = self.cri_perceptual(self.output, self.gt)
            if l_percep is not None:
                l_total += l_percep
                loss_dict['l_percep'] = l_percep
            if l_style is not None:
                l_total += l_style
                loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def test(self):
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                self.output = self.net_g_ema(self.lq)
        else:
            self.net_g.eval()
            with torch.no_grad():
                self.output = self.net_g(self.lq)
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        if self.opt['rank'] == 0:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def nondist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        use_pbar = self.opt['val'].get('pbar', False)

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {metric: 0 for metric in self.opt['val']['metrics'].keys()}
            # initialize the best metric results for each dataset_name (supporting multiple validation datasets)
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        if with_metrics:
            self.metric_results = {metric: 0 for metric in self.metric_results}

        metric_data = dict()
        if use_pbar:
            pbar = tqdm(total=len(dataloader), unit='image')

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]
            self.feed_data(val_data)
            self.test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']])
            metric_data['img'] = sr_img
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']])
                metric_data['img2'] = gt_img
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                if self.opt['is_train']:
                    save_img_path = osp.join(self.opt['path']['visualization'], img_name,
                                             f'{img_name}_{current_iter}.png')
                else:
                    if self.opt['val']['suffix']:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["val"]["suffix"]}.png')
                    else:
                        save_img_path = osp.join(self.opt['path']['visualization'], dataset_name,
                                                 f'{img_name}_{self.opt["name"]}.png')
                imwrite(sr_img, save_img_path)

            if with_metrics:
                # calculate metrics
                for name, opt_ in self.opt['val']['metrics'].items():
                    self.metric_results[name] += calculate_metric(metric_data, opt_)
            if use_pbar:
                pbar.update(1)
                pbar.set_description(f'Test {img_name}')
        if use_pbar:
            pbar.close()

        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= (idx + 1)
                # update the best metric result
                self._update_best_metric_result(dataset_name, metric, self.metric_results[metric], current_iter)

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
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if hasattr(self, 'net_g_ema'):
            self.save_network([self.net_g, self.net_g_ema], 'net_g', current_iter, param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
