import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    module = "hakuir." + module  
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate(obj):
    if isinstance(obj, str):
        return get_obj_from_str(obj)
    return obj
