import os
import time
import torch
from collections import OrderedDict
from copy import deepcopy
from torch.nn.parallel import DataParallel, DistributedDataParallel

from basicsr.models import lr_scheduler as lr_scheduler
from basicsr.utils import get_root_logger
from basicsr.utils.dist_util import master_only


class BaseModel():
    """Base model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda' if opt['num_gpu'] != 0 else 'cpu')
        self.is_train = opt['is_train']
        self.schedulers = []
        self.optimizers = []

    def feed_data(self, data):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        pass

    def save(self, epoch, current_iter):
        """Save networks and training state."""
        pass

    def validation(self, dataloader, current_iter, tb_logger, save_img=False):
        """Validation function.

        Args:
            dataloader (torch.utils.data.DataLoader): Validation dataloader.
            current_iter (int): Current iteration.
            tb_logger (tensorboard logger): Tensorboard logger.
            save_img (bool): Whether to save images. Default: False.
        """
        if self.opt['dist']:
            self.dist_validation(dataloader, current_iter, tb_logger, save_img)
        else:
            self.nondist_validation(dataloader, current_iter, tb_logger, save_img)

    def _initialize_best_metric_results(self, dataset_name):
        """Initialize the best metric results dict for recording the best metric value and iteration."""
        if hasattr(self, 'best_metric_results') and dataset_name in self.best_metric_results:
            return
        elif not hasattr(self, 'best_metric_results'):
            self.best_metric_results = dict()

        # add a dataset record
        record = dict()
        for metric, content in self.opt['val']['metrics'].items():
            better = content.get('better', 'higher')
            init_val = float('-inf') if better == 'higher' else float('inf')
            record[metric] = dict(better=better, val=init_val, iter=-1)
        self.best_metric_results[dataset_name] = record

    def _update_best_metric_result(self, dataset_name, metric, val, current_iter):
        if self.best_metric_results[dataset_name][metric]['better'] == 'higher':
            if val >= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter
        else:
            if val <= self.best_metric_results[dataset_name][metric]['val']:
                self.best_metric_results[dataset_name][metric]['val'] = val
                self.best_metric_results[dataset_name][metric]['iter'] = current_iter

    def model_ema(self, decay=0.999):
        net_g = self.get_bare_model(self.net_g)

        net_g_params = dict(net_g.named_parameters())
        net_g_ema_params = dict(self.net_g_ema.named_parameters())

        for k in net_g_ema_params.keys():
            net_g_ema_params[k].data.mul_(decay).add_(net_g_params[k].data, alpha=1 - decay)

    def get_current_log(self):
        return self.log_dict

    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        if self.opt['dist']:
            find_unused_parameters = self.opt.get('find_unused_parameters', False)
            net = DistributedDataParallel(
                net, device_ids=[torch.cuda.current_device()], find_unused_parameters=find_unused_parameters)
        elif self.opt['num_gpu'] > 1:
            net = DataParallel(net)
        return net

    def get_optimizer(self, optim_type, params, lr, **kwargs):
        if optim_type == 'Adam':
            optimizer = torch.optim.Adam(params, lr, **kwargs)
        else:
            raise NotImplementedError(f'optimizer {optim_type} is not supperted yet.')
        return optimizer

    def setup_schedulers(self):
        def setup_schedulers(self):
            """Set up schedulers."""
            train_opt = self.opt['train']
            print("Current training options in setup_schedulers:", train_opt)  # 打印确认配置

            if 'scheduler' not in train_opt:
                raise KeyError("Scheduler configuration 'scheduler' not found in training options.")

            if 'type' not in train_opt['scheduler']:
                raise KeyError("Scheduler configuration 'type' not found in 'scheduler'.")

            scheduler_type = train_opt['scheduler'].pop('type')
            print("Scheduler type retrieved:", scheduler_type)  # 确认调度器类型

            if scheduler_type in ['MultiStepLR', 'MultiStepRestartLR']:
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.MultiStepLR(optimizer, **train_opt['scheduler']))
            elif scheduler_type == 'CosineAnnealingRestartLR':
                for optimizer in self.optimizers:
                    self.schedulers.append(lr_scheduler.CosineAnnealingRestartLR(optimizer, **train_opt['scheduler']))
            else:
                raise NotImplementedError(f'Scheduler {scheduler_type} is not implemented yet.')

    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net = net.module
        return net

    @master_only
    def print_network(self, net):
        """Print the str and parameter number of a network.

        Args:
            net (nn.Module)
        """
        if isinstance(net, (DataParallel, DistributedDataParallel)):
            net_cls_str = f'{net.__class__.__name__} - {net.module.__class__.__name__}'
        else:
            net_cls_str = f'{net.__class__.__name__}'

        net = self.get_bare_model(net)
        net_str = str(net)
        net_params = sum(map(lambda x: x.numel(), net.parameters()))

        logger = get_root_logger()
        logger.info(f'Network: {net_cls_str}, with parameters: {net_params:,d}')
        logger.info(net_str)

    def _set_lr(self, lr_groups_l):
        """Set learning rate for warmup.

        Args:
            lr_groups_l (list): List for lr_groups, each for an optimizer.
        """
        for optimizer, lr_groups in zip(self.optimizers, lr_groups_l):
            for param_group, lr in zip(optimizer.param_groups, lr_groups):
                param_group['lr'] = lr

    def _get_init_lr(self):
        """Get the initial lr, which is set by the scheduler.
        """
        init_lr_groups_l = []
        for optimizer in self.optimizers:
            init_lr_groups_l.append([v['initial_lr'] for v in optimizer.param_groups])
        return init_lr_groups_l

    def update_learning_rate(self, current_iter, warmup_iter=-1):
        """Update learning rate.

        Args:
            current_iter (int): Current iteration.
            warmup_iter (int)： Warmup iter numbers. -1 for no warmup.
                Default： -1.
        """
        if current_iter > 1:
            for scheduler in self.schedulers:
                scheduler.step()
        # set up warm-up learning rate
        if current_iter < warmup_iter:
            # get initial lr for each group
            init_lr_g_l = self._get_init_lr()
            # modify warming-up learning rates
            # currently only support linearly warm up
            warm_up_lr_l = []
            for init_lr_g in init_lr_g_l:
                warm_up_lr_l.append([v / warmup_iter * current_iter for v in init_lr_g])
            # set learning rate
            self._set_lr(warm_up_lr_l)

    def get_current_learning_rate(self):
        if not self.optimizers:
            raise RuntimeError("No optimizers initialized. Unable to get learning rate.")
        return [param_group['lr'] for param_group in self.optimizers[0].param_groups]

    @master_only
    def save_network(self, net, net_label, current_iter, param_key='params'):
        """Save networks.

        Args:
            net (nn.Module | list[nn.Module]): Network(s) to be saved.
            net_label (str): Network label.
            current_iter (int): Current iter number.
            param_key (str | list[str]): The parameter key(s) to save network.
                Default: 'params'.
        """
        if current_iter == -1:
            current_iter = 'latest'
        save_filename = f'{net_label}_{current_iter}.pth'
        save_path = os.path.join(self.opt['path']['models'], save_filename)

        net = net if isinstance(net, list) else [net]
        param_key = param_key if isinstance(param_key, list) else [param_key]
        assert len(net) == len(param_key), 'The lengths of net and param_key should be the same.'

        save_dict = {}
        for net_, param_key_ in zip(net, param_key):
            net_ = self.get_bare_model(net_)
            state_dict = net_.state_dict()
            for key, param in state_dict.items():
                if key.startswith('module.'):  # remove unnecessary 'module.'
                    key = key[7:]
                state_dict[key] = param.cpu()
            save_dict[param_key_] = state_dict

        # avoid occasional writing errors
        retry = 3
        while retry > 0:
            try:
                torch.save(save_dict, save_path)
            except Exception as e:
                logger = get_root_logger()
                logger.warning(f'Save model error: {e}, remaining retry times: {retry - 1}')
                time.sleep(1)
            else:
                break
            finally:
                retry -= 1
        if retry == 0:
            logger.warning(f'Still cannot save {save_path}. Just ignore it.')
            # raise IOError(f'Cannot save {save_path}.')

    def _print_different_keys_loading(self, crt_net, load_net, strict=True):
        """Print keys with different name or different size when loading models.

        1. Print keys with different names.
        2. If strict=False, print the same key but with different tensor size.
            It also ignore these keys with different sizes (not load).

        Args:
            crt_net (torch model): Current network.
            load_net (dict): Loaded network.
            strict (bool): Whether strictly loaded. Default: True.
        """
        crt_net = self.get_bare_model(crt_net)
        crt_net = crt_net.state_dict()
        crt_net_keys = set(crt_net.keys())
        load_net_keys = set(load_net.keys())

        logger = get_root_logger()
        if crt_net_keys != load_net_keys:
            logger.warning('Current net - loaded net:')
            for v in sorted(list(crt_net_keys - load_net_keys)):
                logger.warning(f'  {v}')
            logger.warning('Loaded net - current net:')
            for v in sorted(list(load_net_keys - crt_net_keys)):
                logger.warning(f'  {v}')

        # check the size for the same keys
        if not strict:
            common_keys = crt_net_keys & load_net_keys
            for k in common_keys:
                if crt_net[k].size() != load_net[k].size():
                    logger.warning(f'Size different, ignore [{k}]: crt_net: '
                                   f'{crt_net[k].shape}; load_net: {load_net[k].shape}')
                    load_net[k + '.ignore'] = load_net.pop(k)

    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        logger = get_root_logger()
        net = self.get_bare_model(net)
        load_net = torch.load(load_path, map_location=lambda storage, loc: storage)
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
                logger.info('Loading: params_ema does not exist, use params.')
            load_net = load_net[param_key]
        logger.info(f'Loading {net.__class__.__name__} model from {load_path}, with param key: [{param_key}].')
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        self._print_different_keys_loading(net, load_net, strict)
        net.load_state_dict(load_net, strict=strict)

    @master_only
    def save_training_state(self, epoch, current_iter):
        """Save training states during training, which will be used for
        resuming.

        Args:
            epoch (int): Current epoch.
            current_iter (int): Current iteration.
        """
        if current_iter != -1:
            state = {'epoch': epoch, 'iter': current_iter, 'optimizers': [], 'schedulers': []}
            for o in self.optimizers:
                state['optimizers'].append(o.state_dict())
            for s in self.schedulers:
                state['schedulers'].append(s.state_dict())
            save_filename = f'{current_iter}.state'
            save_path = os.path.join(self.opt['path']['training_states'], save_filename)

            # avoid occasional writing errors
            retry = 3
            while retry > 0:
                try:
                    torch.save(state, save_path)
                except Exception as e:
                    logger = get_root_logger()
                    logger.warning(f'Save training state error: {e}, remaining retry times: {retry - 1}')
                    time.sleep(1)
                else:
                    break
                finally:
                    retry -= 1
            if retry == 0:
                logger.warning(f'Still cannot save {save_path}. Just ignore it.')
                # raise IOError(f'Cannot save {save_path}.')

    def debug_optimizer_mismatch(self, resume_state):
        """Debug helper to understand optimizer mismatch"""
        import torch
        from basicsr.utils import get_root_logger
        logger = get_root_logger()

        logger.info("\n" + "=" * 70)
        logger.info("OPTIMIZER DEBUG INFO")
        logger.info("=" * 70)

        # 1. 检查当前优化器
        logger.info(f"\nCurrent optimizers: {len(self.optimizers)}")
        for i, opt in enumerate(self.optimizers):
            logger.info(f"\nOptimizer {i}:")
            logger.info(f"  Type: {type(opt).__name__}")
            logger.info(f"  Param groups: {len(opt.param_groups)}")
            total_params = sum(len(g['params']) for g in opt.param_groups)
            logger.info(f"  Total params: {total_params}")
            logger.info(f"  Learning rate: {opt.param_groups[0]['lr']}")

        # 2. 检查 checkpoint 优化器
        resume_optimizers = resume_state.get('optimizers', [])
        logger.info(f"\nCheckpoint optimizers: {len(resume_optimizers)}")
        for i, opt_state in enumerate(resume_optimizers):
            logger.info(f"\nCheckpoint Optimizer {i}:")
            logger.info(f"  Param groups: {len(opt_state['param_groups'])}")
            total_params = sum(len(g['params']) for g in opt_state['param_groups'])
            logger.info(f"  Total params: {total_params}")
            logger.info(f"  Learning rate: {opt_state['param_groups'][0]['lr']}")

        # 3. 详细比较第一个优化器
        if len(resume_optimizers) > 0:
            logger.info("\n" + "-" * 70)
            logger.info("DETAILED COMPARISON - Optimizer 0")
            logger.info("-" * 70)

            current_opt = self.optimizers[0]
            resume_opt = resume_optimizers[0]

            logger.info(f"\nCurrent optimizer 0:")
            for j, group in enumerate(current_opt.param_groups):
                logger.info(f"  Group {j}: {len(group['params'])} params")
                if len(group['params']) > 0:
                    first_param = group['params'][0]
                    logger.info(f"    First param shape: {first_param.shape}")

            logger.info(f"\nCheckpoint optimizer 0:")
            for j, group in enumerate(resume_opt['param_groups']):
                logger.info(f"  Group {j}: {len(group['params'])} params")

        # 4. 检查 schedulers
        logger.info("\n" + "-" * 70)
        logger.info("SCHEDULER INFO")
        logger.info("-" * 70)
        logger.info(f"Current schedulers: {len(self.schedulers)}")
        logger.info(f"Checkpoint schedulers: {len(resume_state.get('schedulers', []))}")

        # 5. 检查网络参数
        logger.info("\n" + "-" * 70)
        logger.info("NETWORK PARAMETERS")
        logger.info("-" * 70)

        def count_parameters(model):
            return sum(p.numel() for p in model.parameters() if p.requires_grad)

        logger.info(f"net_g parameters: {count_parameters(self.net_g):,}")
        if hasattr(self, 'lpg'):
            logger.info(f"lpg parameters: {count_parameters(self.lpg):,}")

        logger.info("=" * 70 + "\n")

    def analyze_parameter_difference(self, checkpoint_or_path):
        """详细分析当前模型和 checkpoint 的参数差异

        Args:
            checkpoint_or_path: 可以是：
                - 字符串：模型文件路径
                - 字典：已加载的 checkpoint
        """
        import torch
        from basicsr.utils import get_root_logger
        import os.path as osp

        logger = get_root_logger()

        logger.info("\n" + "=" * 80)
        logger.info("DETAILED PARAMETER ANALYSIS")
        logger.info("=" * 80)

        # 1. 确定如何加载 checkpoint
        if isinstance(checkpoint_or_path, str):
            # 传入的是文件路径
            if not osp.exists(checkpoint_or_path):
                logger.error(f"Checkpoint file not found: {checkpoint_or_path}")
                return None

            logger.info(f"Loading checkpoint from: {checkpoint_or_path}")
            checkpoint = torch.load(checkpoint_or_path, map_location='cpu')

        elif isinstance(checkpoint_or_path, dict):
            # 传入的是已加载的字典
            checkpoint = checkpoint_or_path
            logger.info("Using provided checkpoint dictionary")

            # 如果是 training state，需要加载对应的模型文件
            if 'params' not in checkpoint and 'iter' in checkpoint:
                logger.info("Detected training state, loading model file...")
                current_iter = checkpoint.get('iter', 0)
                model_path = osp.join(self.opt['path']['models'], f'net_g_{current_iter}.pth')

                if osp.exists(model_path):
                    logger.info(f"Loading model from: {model_path}")
                    checkpoint = torch.load(model_path, map_location='cpu')
                else:
                    logger.error(f"Model file not found: {model_path}")
                    return None
        else:
            logger.error(f"Invalid checkpoint type: {type(checkpoint_or_path)}")
            return None

        # 2. 检查 checkpoint 结构
        logger.info(f"\nCheckpoint keys: {list(checkpoint.keys())}")

        # 3. 获取参数字典
        if 'params' in checkpoint:
            checkpoint_params = checkpoint['params']
            logger.info("Using 'params' from checkpoint")
        elif 'params_ema' in checkpoint:
            checkpoint_params = checkpoint['params_ema']
            logger.info("Using 'params_ema' from checkpoint")
        elif 'state_dict' in checkpoint:
            checkpoint_params = checkpoint['state_dict']
            logger.info("Using 'state_dict' from checkpoint")
        else:
            # 假设 checkpoint 本身就是 state_dict
            if any(isinstance(v, torch.Tensor) for v in checkpoint.values()):
                checkpoint_params = checkpoint
                logger.info("Checkpoint is a direct state_dict")
            else:
                logger.error("Cannot find parameters in checkpoint!")
                logger.error(f"Available keys: {list(checkpoint.keys())}")
                return None

        current_params = self.net_g.state_dict()

        logger.info(f"\nCheckpoint 参数层数: {len(checkpoint_params)}")
        logger.info(f"Current 参数层数:    {len(current_params)}")

        # 4. 详细对比
        same_layers = []
        shape_mismatch = []
        only_in_current = []
        only_in_checkpoint = []

        # 检查当前模型的每一层
        for key in sorted(current_params.keys()):
            current_shape = current_params[key].shape
            current_numel = current_params[key].numel()

            if key in checkpoint_params:
                checkpoint_shape = checkpoint_params[key].shape
                checkpoint_numel = checkpoint_params[key].numel()

                if current_shape == checkpoint_shape:
                    same_layers.append((key, current_shape, current_numel))
                else:
                    shape_mismatch.append({
                        'key': key,
                        'current_shape': current_shape,
                        'checkpoint_shape': checkpoint_shape,
                        'current_numel': current_numel,
                        'checkpoint_numel': checkpoint_numel,
                        'diff_numel': current_numel - checkpoint_numel
                    })
            else:
                only_in_current.append((key, current_shape, current_numel))

        # 检查只在 checkpoint 中的层
        for key in checkpoint_params.keys():
            if key not in current_params:
                only_in_checkpoint.append((key, checkpoint_params[key].shape,
                                           checkpoint_params[key].numel()))

        # 5. 打印统计
        logger.info("\n" + "-" * 80)
        logger.info("COMPARISON SUMMARY")
        logger.info("-" * 80)
        logger.info(f"✓ 完全相同的层: {len(same_layers)}")
        logger.info(f"⚠ 形状不匹配:   {len(shape_mismatch)}")
        logger.info(f"➕ 新增的层:     {len(only_in_current)}")
        logger.info(f"➖ 移除的层:     {len(only_in_checkpoint)}")

        # 6. 详细打印形状不匹配
        if shape_mismatch:
            logger.info("\n" + "=" * 80)
            logger.info("⚠ SHAPE MISMATCH DETAILS")
            logger.info("=" * 80)

            total_diff = 0
            for item in shape_mismatch:
                logger.info(f"\n📌 {item['key']}")
                logger.info(f"   Current:    {item['current_shape']} = {item['current_numel']:,} params")
                logger.info(f"   Checkpoint: {item['checkpoint_shape']} = {item['checkpoint_numel']:,} params")
                logger.info(f"   Difference: {item['diff_numel']:+,d} params")
                total_diff += item['diff_numel']

            logger.info(f"\n📊 总参数差异: {total_diff:+,d}")

        # 7. 新增的层
        if only_in_current:
            logger.info("\n" + "=" * 80)
            logger.info("➕ NEW LAYERS")
            logger.info("=" * 80)

            total_new = 0
            for key, shape, numel in only_in_current[:10]:  # 只显示前 10 个
                logger.info(f"   {key}: {shape} ({numel:,} params)")
                total_new += numel

            if len(only_in_current) > 10:
                logger.info(f"   ... and {len(only_in_current) - 10} more")
                for key, shape, numel in only_in_current[10:]:
                    total_new += numel

            logger.info(f"\n📊 新增总参数: {total_new:,}")

        # 8. 移除的层
        if only_in_checkpoint:
            logger.info("\n" + "=" * 80)
            logger.info("➖ REMOVED LAYERS")
            logger.info("=" * 80)

            total_removed = 0
            for key, shape, numel in only_in_checkpoint[:10]:
                logger.info(f"   {key}: {shape} ({numel:,} params)")
                total_removed += numel

            if len(only_in_checkpoint) > 10:
                logger.info(f"   ... and {len(only_in_checkpoint) - 10} more")
                for key, shape, numel in only_in_checkpoint[10:]:
                    total_removed += numel

            logger.info(f"\n📊 移除总参数: {total_removed:,}")

        # 9. 分析前几层
        logger.info("\n" + "=" * 80)
        logger.info("🔍 FIRST 5 LAYERS ANALYSIS")
        logger.info("=" * 80)

        for i, key in enumerate(sorted(current_params.keys())[:5]):
            logger.info(f"\n[{i + 1}] {key}")
            logger.info(f"    Current: {current_params[key].shape}")

            if key in checkpoint_params:
                logger.info(f"    Checkpoint: {checkpoint_params[key].shape}")

                # 分析卷积层
                if len(current_params[key].shape) == 4:
                    c_out, c_in, h, w = current_params[key].shape
                    logger.info(f"    → Conv: out_ch={c_out}, in_ch={c_in}, kernel={h}x{w}")

                    if key in checkpoint_params:
                        ckpt_out, ckpt_in, ckpt_h, ckpt_w = checkpoint_params[key].shape
                        if c_in != ckpt_in:
                            logger.warning(f"    ⚠️ 输入通道变化: {ckpt_in} → {c_in}")
                        if c_out != ckpt_out:
                            logger.warning(f"    ⚠️ 输出通道变化: {ckpt_out} → {c_out}")
            else:
                logger.warning(f"    ⚠️ 此层不在 checkpoint 中")

        logger.info("\n" + "=" * 80 + "\n")

        return {
            'same': len(same_layers),
            'mismatch': len(shape_mismatch),
            'new': len(only_in_current),
            'removed': len(only_in_checkpoint),
            'details': {
                'shape_mismatch': shape_mismatch,
                'new_layers': only_in_current,
                'removed_layers': only_in_checkpoint
            }
        }

    def resume_training(self, resume_state):
        """Reload the optimizers and schedulers for resumed training.

        Args:
            resume_state (dict): Resume state.
        """


        from basicsr.utils import get_root_logger
        logger = get_root_logger()

        analysis = self.analyze_parameter_difference(resume_state)
        self.debug_optimizer_mismatch(resume_state)

        resume_optimizers = resume_state.get('optimizers', [])
        resume_schedulers = resume_state.get('schedulers', [])

        # ==================== 优化器恢复 ====================
        current_opt_count = len(self.optimizers)
        resume_opt_count = len(resume_optimizers)

        if resume_opt_count != current_opt_count:
            logger.warning('=' * 70)
            logger.warning(f'Optimizer count mismatch!')
            logger.warning(f'  Checkpoint: {resume_opt_count} optimizer(s)')
            logger.warning(f'  Current:    {current_opt_count} optimizer(s)')
            logger.warning('=' * 70)

            # 只加载匹配数量的优化器
            num_opt_to_load = min(resume_opt_count, current_opt_count)

            for i in range(num_opt_to_load):
                try:
                    self.optimizers[i].load_state_dict(resume_optimizers[i])
                    logger.info(f'✓ Successfully loaded optimizer {i}')
                except Exception as e:
                    logger.warning(f'✗ Failed to load optimizer {i}: {type(e).__name__}')
                    logger.warning(f'  Optimizer {i} will use fresh initialization')

            # 新增的优化器保持初始化状态（不降低学习率）
            if current_opt_count > resume_opt_count:
                logger.info(
                    f'ℹ New optimizers ({resume_opt_count} to {current_opt_count - 1}) use config learning rates')

        else:
            # 数量匹配，正常加载
            logger.info(f'Loading {current_opt_count} optimizer(s)...')
            for i, resume_opt in enumerate(resume_optimizers):
                try:
                    self.optimizers[i].load_state_dict(resume_opt)
                    logger.info(f'✓ Optimizer {i} loaded')
                except Exception as e:
                    logger.warning(f'✗ Failed to load optimizer {i}, using fresh initialization')

        # ==================== 调度器恢复 ====================
        current_sch_count = len(self.schedulers)
        resume_sch_count = len(resume_schedulers)

        if resume_sch_count != current_sch_count:
            logger.warning(f'Scheduler count mismatch (checkpoint: {resume_sch_count}, current: {current_sch_count})')
            num_sch_to_load = min(resume_sch_count, current_sch_count)
            for i in range(num_sch_to_load):
                try:
                    self.schedulers[i].load_state_dict(resume_schedulers[i])
                    logger.info(f'✓ Scheduler {i} loaded')
                except:
                    logger.warning(f'✗ Scheduler {i} uses fresh initialization')
        else:
            for i, resume_sch in enumerate(resume_schedulers):
                try:
                    self.schedulers[i].load_state_dict(resume_sch)
                    logger.info(f'✓ Scheduler {i} loaded')
                except:
                    logger.warning(f'✗ Scheduler {i} uses fresh initialization')

        # ==================== 恢复训练进度 ====================
        self.start_epoch = resume_state.get('epoch', 0)
        if hasattr(self, 'current_iter'):
            self.current_iter = resume_state.get('iter', 0)

        logger.info('=' * 70)
        logger.info(f'✓ Resumed from epoch {self.start_epoch}')
        if hasattr(self, 'current_iter'):
            logger.info(f'  Iteration: {self.current_iter}')
        logger.info('=' * 70)

    def reduce_loss_dict(self, loss_dict):
        """reduce loss dict.

        In distributed training, it averages the losses among different GPUs .

        Args:
            loss_dict (OrderedDict): Loss dict.
        """
        with torch.no_grad():
            if self.opt['dist']:
                keys = []
                losses = []
                for name, value in loss_dict.items():
                    keys.append(name)
                    losses.append(value)
                losses = torch.stack(losses, 0)
                torch.distributed.reduce(losses, dst=0)
                if self.opt['rank'] == 0:
                    losses /= self.opt['world_size']
                loss_dict = {key: loss for key, loss in zip(keys, losses)}

            log_dict = OrderedDict()
            for name, value in loss_dict.items():
                log_dict[name] = value.mean().item()

            return log_dict
