import numpy as np

class Config:
    # basic parameters
    out_reduce = 0.001
    exemplar_sz = 127
    instance_sz = 255
    context = 0.5
    # inference parameters
    scale_num = 3
    scale_step = 1.0375
    scale_lr = 0.59
    scale_penalty = 0.9745
    window_influence = 0.176
    response_sz = 17
    response_up = 16
    total_stride = 8
    # train parameters
    epoch_num = 50
    batch_size = 8
    num_workers = 2
    initial_lr = 1e-2
    ultimate_lr = 1e-5
    weight_decay = 5e-4
    momentum = 0.9
    r_pos = 16
    r_neg = 0


cfg = Config()
a = cfg.batch_size