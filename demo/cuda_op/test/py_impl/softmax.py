import torch


def base_softmax(x, dim):
    """base version"""
    x_exp_sum = x.exp().sum(dim)
    res = x.exp() / x_exp_sum
    return res


def safe_softmax(x, dim):
    x_max = x.max(dim, keepdim=True)[0]
    exp = torch.exp(x - x_max)
    exp_sum = torch.sum(exp, dim)
    res = exp / exp_sum
    return res


# def online_softmax(x, dim):
#     l_prev = 0
#     m_prev = -999999
#     shapes = x.shapes()
#     shape = shapes[dim]
#     blk_size = 2
#     blk_cnt = shape / blk_size
#     for blk_idx in range(blk_cnt):
#         x_blk = x
#         x_blk_max = x_blk.max(dim)
