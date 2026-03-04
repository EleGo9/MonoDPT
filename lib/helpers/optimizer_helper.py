import math
import torch
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from lib.helpers.config_helper import Config, BackboneType

def get_paramgroups_monodpt_resnet(cfg, model):
    weights, biases = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if 'bias' in name:
            biases += [param]
        else:
            weights += [param]

    param_groups = [
        {'name': 'biases', 'params': biases, 'weight_decay': 0},
        {'name': 'weights', 'params': weights, 'weight_decay': cfg.optimizer.weight_decay}
    ]  # mod, added "name" field
    return param_groups

def get_paramgroups_monodpt_classic(cfg, model):
    backbone, other = [], []
    for name, param in model.named_parameters():
        if 'backbone' in name:
            backbone += [param]
        else:
            other += [param]
    param_groups = [
        {'name': 'backbone', 'params': backbone, 'lr': cfg.optimizer.lr, 'weight_decay': 0.01, "lr_scale": 1.0},
        {'name': 'default', 'params': other, 'lr': cfg.optimizer.lr, 'weight_decay': cfg.optimizer.weight_decay,
         "lr_scale": 10.0}
    ]

    return param_groups


def get_paramgroups_monodpt(cfg, model):
    layer_decay = 0.9

    def get_layer_id(name: str):
        if any(
            name.startswith(p)
            for p in [
                "backbone.0.backbone.cls_token",
                "backbone.0.backbone.pos_embed",
                "backbone.0.backbone.mask_token",
                "backbone.0.backbone.patch_embed",
            ]
        ):
            return 0
        elif "backbone.0.backbone.blocks" in name:
            try:
                # Extract block index robustly
                idx = int(name.split("backbone.0.backbone.blocks.")[1].split(".")[0])
                return idx + 1
            except (IndexError, ValueError):
                return 0
        else:
            return None  # not a recognized backbone layer

    param_groups = {}
    num_layers = len(model.backbone[0].backbone.blocks)
    layer_scales = list(
        layer_decay ** (num_layers - i) for i in range(num_layers + 1)
    )

    no_weight_decay = [
        "backbone.0.backbone.pos_embed",
        "backbone.0.backbone.cls_token",  # TODO: Add task tokens
        "backbone.0.backbone.mask_token",
    ]

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        if name.startswith("backbone.0.backbone"):
            if (
                param.ndim == 1
                or name in no_weight_decay
                or name.endswith(".bias")
                or "norm" in name
                or "gamma" in name
            ):
                g_decay = "no_decay"
                this_decay = 0.0
            else:
                g_decay = "decay"
                this_decay = cfg.optimizer.weight_decay

            layer_id = get_layer_id(name)
            if layer_id is None:
                group_name = f"backbone_other_{g_decay}"
                this_scale = 1.0
            else:
                group_name = f"backbone_layer_{layer_id}_{g_decay}"
                this_scale = layer_scales[layer_id]

        # wip: scale for DPT head
        elif name.startswith("backbone.0.dpt_head"):
            if (
                param.ndim == 1
                or name in no_weight_decay
                or name.endswith(".bias")
                or "norm" in name
                or "gamma" in name
            ):
                g_decay = "no_decay"
                this_decay = 0.0
            else:
                g_decay = "decay"
                this_decay = cfg.optimizer.weight_decay

            group_name = f"dtp_head_{g_decay}"
            this_scale = 1.0

        else:
            if name.endswith(".bias") or "norm" in name or "gamma" in name:
                group_name = "other_nodecay"
                this_scale = 10.0
                this_decay = 0.0
            else:
                group_name = "other"
                this_scale = 10.0
                this_decay = cfg.optimizer.weight_decay

        if group_name not in param_groups:
            param_groups[group_name] = {
                "lr_scale": this_scale,
                "weight_decay": this_decay,
                "name": group_name,
                "params": [],
            }

        assert not isinstance(param, str)
        param_groups[group_name]["params"].append(param)

    return list(param_groups.values())

def build_optimizer(cfg: Config, model):

    # todo: fare meglio
    if cfg.model.backbone == BackboneType.resnet50:
        param_groups = get_paramgroups_monodpt_resnet(cfg, model)
    else:
        param_groups = get_paramgroups_monodpt(cfg, model)

    if cfg.optimizer.type == 'sgd':
        optimizer = optim.SGD(param_groups, lr=cfg.optimizer.lr, momentum=0.9)
    elif cfg.optimizer.type == 'adam':
        optimizer = optim.Adam(param_groups, lr=cfg.optimizer.lr)
    elif cfg.optimizer.type == 'adamw':
        optimizer = AdamW(param_groups, lr=cfg.optimizer.lr)
    #elif cfg.optimizer.type == 'adamw_depthany':
    #    optimizer = AdamW(param_groups, lr=cfg.optimizer.lr, betas=(0.9, 0.999), weight_decay=0.01)
    else:
        raise NotImplementedError("%s optimizer is not supported" % cfg.optimizer.type)

    return optimizer


class AdamW(Optimizer):
    """Implements Adam algorithm.
    It has been proposed in `Adam: A Method for Stochastic Optimization`_.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
    .. _Adam\: A Method for Stochastic Optimization:
        https://arxiv.org/abs/1412.6980
    .. _On the Convergence of Adam and Beyond:
        https://openreview.net/forum?id=ryQu7f-RZ
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)
        super(AdamW, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                # if group['weight_decay'] != 0:
                #     grad = grad.add(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])

                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                # p.data.addcdiv_(-step_size, exp_avg, denom)
                p.data.add_(-step_size,  torch.mul(p.data, group['weight_decay']).addcdiv_(1, exp_avg, denom))

        return loss
