import torch
import math
import cuda_op
import torch.nn as nn
import torch.nn.functional as F


def attention(q, k, v):
    att = (q @ k.transpose(-2, -1) * (1.0 / math.sqrt(k.size(-1))))
    att = F.softmax(att, dim=-1)
    y = att @ v
    return y


def test_basic():
    batch_size = 16
    n_head = 12
    seq_len = 64
    head_embd = 64

    q = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    k = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()
    v = torch.randn(batch_size, n_head, seq_len, head_embd).cuda()

    golden_o = attention(q, k, v).cpu()
    actual_o = cuda_op.flash_attention(q, k, v).cpu()
    torch.testing.assert_close(golden_o, actual_o, atol=1e-5, rtol=1e-5)
