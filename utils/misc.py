import os
import torch

def create_path(path):
    os.makedirs(path, exist_ok=True)

def calc_model_size(model):
    param_size = sum(p.nelement() for p in model.parameters())
    buffer_size = sum(b.nelement() for b in model.buffers())
    return (param_size + buffer_size) / 1024 ** 2