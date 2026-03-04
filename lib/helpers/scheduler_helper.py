import torch
import torch.nn as nn
from torch.optim import Optimizer
import torch.optim.lr_scheduler as lr_sched
import math

from typing import Optional
from lib.helpers.config_helper import Config


def build_lr_scheduler(cfg: Config,
                       optimizer: torch.optim.Optimizer,
                       max_epochs: int,
                       last_epoch: int,
                       steps_per_epoch: int):

    # Build warmup scheduler
    if cfg.lr_scheduler.warmup is not None:
        # Support warmup specified in epochs; convert to steps
        warmup_epochs = cfg.lr_scheduler.warmup_epochs
        warmup_steps = warmup_epochs * steps_per_epoch
        init_lr = cfg.lr_scheduler.warmup_init_lr
        if cfg.lr_scheduler.warmup == "linear":
            # linear warmup code
            warmup_scheduler = LinearWarmupLR(optimizer, num_epoch=warmup_steps, init_lr=init_lr)
        elif cfg.lr_scheduler.warmup == "cos":
            # cos warmup code
            warmup_scheduler = CosineWarmupLR(optimizer, num_epoch=warmup_steps, init_lr=init_lr)
        else:
            warmup_scheduler = None
            # default case
        # match cfg.lr_scheduler.warmup:
        #     case "cos":
        #         warmup_scheduler = CosineWarmupLR(optimizer, num_epoch=warmup_steps, init_lr=init_lr)
        #     case "linear":
        #         warmup_scheduler = LinearWarmupLR(optimizer, num_epoch=warmup_steps, init_lr=init_lr)
        #     case _:
        #         warmup_scheduler = None
    else:
        warmup_steps = 0
        warmup_scheduler = None

    # Build decay scheduler
    num_steps = (max_epochs * steps_per_epoch) - warmup_steps
    if cfg.lr_scheduler.decay == "step":
        decay_list = cfg.lr_scheduler.decay_list
        decay_rate = cfg.lr_scheduler.decay_rate
        decay_scheduler = StepDecayLR(optimizer, decay_list=decay_list,decay_rate=decay_rate,
                                      steps_per_epoch=steps_per_epoch, last_epoch=last_epoch)
    elif cfg.lr_scheduler.decay == "cos":
        min_lr = cfg.lr_scheduler.min_decay_lr
        decay_scheduler = CosineDecayScheduler(optimizer, max_steps=num_steps, min_lr=min_lr, last_epoch=last_epoch)
    elif cfg.lr_scheduler.decay == "poly":
        def poly_decay_lambda(current_step, total_steps, power=0.9): # Depth-anything V2 scheduler
            return (1 - current_step / total_steps) ** power
        decay_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lambda step: poly_decay_lambda(step, num_steps))
    else:
        decay_scheduler = lr_sched.LambdaLR(optimizer, lambda x: 1, last_epoch=last_epoch) # constant lr    
    # match cfg.lr_scheduler.decay:
    #     case "step":
    #         decay_list = cfg.lr_scheduler.decay_list
    #         decay_rate = cfg.lr_scheduler.decay_rate
    #         decay_scheduler = StepDecayLR(optimizer, decay_list=decay_list,decay_rate=decay_rate,
    #                                       steps_per_epoch=steps_per_epoch, last_epoch=last_epoch)
    #     case "cos":
    #         min_lr = cfg.lr_scheduler.min_decay_lr
    #         decay_scheduler = CosineDecayScheduler(optimizer, max_steps=num_steps, min_lr=min_lr, last_epoch=last_epoch)
    #     case "poly":
    #         def poly_decay_lambda(current_step, total_steps, power=0.9): # Depth-anything V2 scheduler
    #             return (1 - current_step / total_steps) ** power
    #         decay_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lambda step: poly_decay_lambda(step, num_steps))
    #     case _:
    #         decay_scheduler = lr_sched.LambdaLR(optimizer, lambda x: 1, last_epoch=last_epoch) # constant lr

    return WarmupThenScheduler(optimizer, warmup_scheduler, decay_scheduler)

class WarmupThenScheduler(lr_sched._LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 warmup_scheduler: Optional[lr_sched._LRScheduler],
                 main_scheduler: lr_sched._LRScheduler):

        self.warmup_scheduler = warmup_scheduler
        self.main_scheduler = main_scheduler
        self.num_warmup = warmup_scheduler.num_epoch if warmup_scheduler is not None else 0
        self.finished_warmup = False
        super().__init__(optimizer)

    def step(self, epoch=None):
        # Use global step count if not provided
        current_step = self.last_epoch + 1 if epoch is None else epoch

        if self.warmup_scheduler and current_step <= self.num_warmup:
            self.warmup_scheduler.last_epoch = current_step
            self.warmup_scheduler.step()
            self.finished_warmup = False
        else:
            if not self.finished_warmup:
                # Sync main scheduler with warmup boundary
                self.main_scheduler.last_epoch = current_step - 1
                self.finished_warmup = True
            self.main_scheduler.step()

        self.last_epoch = current_step

    def get_last_lr(self):
        if self.warmup_scheduler and self.last_epoch <= self.num_warmup:
            return self.warmup_scheduler.get_lr()
        return self.main_scheduler.get_lr()


#-----------------
# DECAY SCHEDULERS
#-----------------
class CosineDecayScheduler(lr_sched._LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        min_lr: float,
        last_epoch: int = -1,
    ):
        self.max_steps = max_steps
        self.min_lr = min_lr
        self.fixed_lrs = []  # todo!

        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch
        return [
            self.min_lr
            + 0.5 * (base_lr - self.min_lr) * (1 + math.cos(math.pi * step / self.max_steps))
            if i not in self.fixed_lrs else base_lr
            for i, base_lr in enumerate(self.base_lrs)
        ]


class StepDecayLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, decay_list, decay_rate, steps_per_epoch=None, last_epoch=-1):
        if steps_per_epoch:
            decay_list_steps = [int(d * steps_per_epoch) for d in decay_list]
        else:
            decay_list_steps = decay_list

        def lr_lbmd(cur_count):
            cur_decay = 1
            for decay_step in decay_list_steps:
                if cur_count >= decay_step:
                    cur_decay *= decay_rate
            return cur_decay

        self.lambda_lr = lr_sched.LambdaLR(optimizer, lr_lbmd, last_epoch=last_epoch)
        super().__init__(optimizer, last_epoch)

    def step(self, epoch=None):
        self.lambda_lr.step(epoch)
        self.last_epoch = self.lambda_lr.last_epoch

    def get_lr(self):
        return self.lambda_lr.get_lr()

#-----------------
# WARMUP SCHEDULERS
#-----------------
class CosineWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        # num_epoch can represent epochs or total warmup steps
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(CosineWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        # Clamp progress so that when used per-step we don't overshoot
        progress = min(self.last_epoch, self.num_epoch) / float(self.num_epoch) if self.num_epoch > 0 else 1.0
        return [self.init_lr + (base_lr - self.init_lr) *
                (1 - math.cos(math.pi * progress)) / 2
                for base_lr in self.base_lrs]


class LinearWarmupLR(lr_sched._LRScheduler):
    def __init__(self, optimizer, num_epoch, init_lr=0.0, last_epoch=-1):
        # num_epoch can represent epochs or total warmup steps
        self.num_epoch = num_epoch
        self.init_lr = init_lr
        super(LinearWarmupLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        progress = min(self.last_epoch, self.num_epoch) / float(self.num_epoch) if self.num_epoch > 0 else 1.0
        return [self.init_lr + (base_lr - self.init_lr) * progress
                for base_lr in self.base_lrs]


"""
def build_bnm_scheduler(cfg, model, last_epoch, steps_per_epoch=None):
    if not cfg.lr_scheduler.enabled:
        return None

    # Convert epoch-based decay list to steps if needed
    if steps_per_epoch:
        decay_list_steps = [int(d * steps_per_epoch) for d in cfg.lr_scheduler.decay_list]
    else:
        decay_list_steps = cfg.lr_scheduler.decay_list

    def bnm_lmbd(cur_count):
        cur_decay = 1
        for decay_step in decay_list_steps:
            if cur_count >= decay_step:
                cur_decay = cur_decay * cfg.lr_scheduler.decay_rate
        return max(cfg.lr_scheduler.momentum * cur_decay, cfg.lr_scheduler.clip)

    bnm_scheduler = BNMomentumScheduler(model, bnm_lmbd, last_epoch=last_epoch)
    return bnm_scheduler


def set_bn_momentum_default(bn_momentum):
    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn


class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError("Class '{}' is not a PyTorch nn Module".format(type(model).__name__))

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        # 'epoch' here can represent epoch index or global step index depending on usage
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))
"""

if __name__ == '__main__':
    import torch
    import torch.optim as optim
    import matplotlib.pyplot as plt
    import math
    from omegaconf import OmegaConf

    # Build Configuration Object
    config_file = OmegaConf.load("../../configs/monodpt.yaml")
    cfg = Config(**config_file)

    # Dummy optimizer
    model = torch.nn.Linear(10, 10)
    optimizer = optim.SGD(
        [
            {"params": [p for n, p in model.named_parameters() if 'bias' in n], "lr": 0.1, "lr_scale": 1.0},  # base group
            {"params": [p for n, p in model.named_parameters() if 'bias' not in n], "lr": 0.1, "lr_scale": 0.5},  # scaled group
        ]
    )

    # -------------------------------------------------------------------
    # Build scheduler
    total_epochs = cfg.trainer.max_epoch
    steps_per_epoch = 256
    scheduler = build_lr_scheduler(cfg, optimizer, total_epochs, last_epoch=-1, steps_per_epoch=steps_per_epoch)

    # -------------------------------------------------------------------
    # Simulate LR over all steps
    lrs_group0 = []
    lrs_group1 = []

    for step in range(total_epochs * steps_per_epoch):
        # Advance scheduler
        scheduler.step()

        # Apply per-group scaling manually
        for lr, group in zip(scheduler.get_last_lr(), optimizer.param_groups):
            scale = group.get("lr_scale", 1.0)
            group["lr"] = lr * scale

        # Record at epoch boundaries (every 256 steps)
        if step % steps_per_epoch == 0:
            lrs_group0.append(optimizer.param_groups[0]["lr"])
            lrs_group1.append(optimizer.param_groups[1]["lr"])

    # -------------------------------------------------------------------
    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(lrs_group0, label="Group 0 (scale=1.0)")
    plt.plot(lrs_group1, label="Group 1 (scale=0.5)")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("LR Schedule with Per-Group Scaling")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("scheduler_scaled.png")
    print("Saved plot to scheduler_scaled.png")