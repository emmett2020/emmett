import torch
import math


def base_softmax(x, dim):
    """base version"""
    x_exp_sum = x.exp().sum(dim)
    res = x.exp() / x_exp_sum
    return res


def safe_softmax(x, dim):
    """avoid overflow"""
    x_max = x.max(dim, keepdim=True)[0]
    exp = torch.exp(x - x_max)
    exp_sum = torch.sum(exp, dim)
    res = exp / exp_sum
    return res


def online_softmax(x, dim):
    """calculate iterately"""
    slices = torch.unbind(x, dim)
    assert len(slices) > 0

    l_prev = torch.zeros_like(slices[0])
    m_prev = torch.full_like(slices[0], -math.inf)

    for data in slices:
        m = torch.max(m_prev, data)
        l = l_prev * torch.exp(m_prev - m) + torch.exp(data - m)
        m_prev = m
        l_prev = l

    results = []
    for data in slices:
        o = torch.exp(data - m_prev) / l_prev
        results.append(o)

    stacked = torch.stack(results, dim)
    return stacked
