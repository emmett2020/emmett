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


def block_online_softmax(x, dim):
    """calculate iterately"""
    blk_size = 2
    shapes = x.shape
    shape = shapes[dim]
    assert shape % blk_size == 0, "The given shape must be divisible by blk_size for simplicity"

    slices = torch.split(x, blk_size, dim)
    assert len(slices) > 0

    lm_shape = list(shapes)
    lm_shape[dim] = 1
    l_prev = torch.zeros(lm_shape)
    m_prev = torch.full(lm_shape, -math.inf)

    for data in slices:
        m_cur = torch.max(data, dim=dim, keepdim=True)[0]
        exp = torch.exp(data - m_cur)
        l_cur = torch.sum(exp, dim=dim)
        m = torch.max(m_prev, m_cur)
        l = l_prev * torch.exp(m_prev - m) + l_cur * torch.exp(m_cur - m)
        m_prev = m
        l_prev = l

    o = torch.exp(x - m_prev) / l_prev
    return o
