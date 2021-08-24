# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Any, Dict, List

import torch
from torch.optim.lr_scheduler import LambdaLR, OneCycleLR

from .lr_scheduler import PolyLR, WarmupCosineLR, WarmupMultiStepLR


def build_optimizer(config, model: torch.nn.Module) -> torch.optim.Optimizer:
    """
    Build an optimizer from config.SOLVER.OPTIMIZER
    """
    if config['NAME'] == "SGD":
        params: List[Dict[str, Any]] = []
        for key, value in model.named_parameters():
            if not config.get("WEIGHT_DECAY_CONV_ONLY", False):
                if not value.requires_grad:
                    continue
                lr = config['BASE_LR']
                weight_decay = config['WEIGHT_DECAY']
                if key.endswith("norm.weight") or key.endswith("norm.bias"):
                    weight_decay = config['WEIGHT_DECAY_NORM']
                elif key.endswith(".bias"):
                    # NOTE: unlike Detectron v1, we now default BIAS_LR_FACTOR to 1.0
                    # and WEIGHT_DECAY_BIAS to WEIGHT_DECAY so that bias optimizer
                    # hyperparameters are by default exactly the same as for regular
                    # weights.
                    lr = config['BASE_LR'] * config['BIAS_LR_FACTOR']
                    weight_decay = config['WEIGHT_DECAY_BIAS']
            else:
                lr = config['BASE_LR']
                if "conv.weight" not in key:
                    weight_decay = 0
                else:
                    weight_decay = config['WEIGHT_DECAY']
            # multiply lr for gating function
            if "GATE_LR_MULTI" in config:
                if config['GATE_LR_MULTI'] > 0.0 and "gate_conv" in key:
                    lr *= config['GATE_LR_MULTI']

            params += [{
                "params": [value],
                "lr": lr,
                "weight_decay": weight_decay
            }]
        optimizer = torch.optim.SGD(params, lr, momentum=config['MOMENTUM'])
    elif config['NAME'] == "AdamW":
        lr = config['BASE_LR']
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=lr,
                                     betas=config['BETAS'],
                                     weight_decay=config['WEIGHT_DECAY'],
                                     amsgrad=config['AMSGRAD'])
    return optimizer

