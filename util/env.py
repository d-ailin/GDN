import torch
import numpy as np

_device = None 

def get_device():
    # return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return _device

def set_device(dev):
    global _device
    _device = dev

def init_work(worker_id, seed):
    np.random.seed(seed + worker_id)
