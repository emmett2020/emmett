import torch
import cuda_op
import torch.nn as nn
import torch.nn.functional as F
import pytest


def test_basic():
    eps = 1e-5

    N = 1
    C = 128
    H = 16
    W = 16
    G = 32
    shape = [N, C, H, W]

    input_t = torch.randn(shape, dtype=torch.float32)
    gamma = torch.ones(C, dtype=torch.float32)
    beta = torch.zeros(C, dtype=torch.float32)

    for group_idx in range(int(C / G)):
        mean = input_t[0, group_idx * G:(group_idx + 1) * G:].mean().item()
        var = input_t[0, group_idx * G:(group_idx + 1) * G:].var().item()
        std = input_t[0, group_idx * G:(group_idx + 1) * G:].std().item()
        rstd = 1 / input_t[0, group_idx * G:(group_idx + 1) * G:].std().item()

        print(
            f"group_idx: {group_idx}, mean: {mean:.6f}, var: {var:.6f} std: {std:.6f} rstd:{rstd:.6f}"
        )

    golden = F.group_norm(input_t, G, gamma, beta, eps)
    input_cuda = input_t.to("cuda")
    actual = cuda_op.group_norm(input_cuda, gamma.to("cuda"), beta.to("cuda"),
                                G, eps).to("cpu")
    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1e-5)


def test_perf():
    eps = 1e-5

    N = 16
    C = 128
    H = 256
    W = 256
    G = 32
    shape = [N, C, H, W]

    therotical_time = (12 * N * C * H * W) / (504 * 10**3)
    print(f"\ntherotical performance: {therotical_time}us")

    input_t = torch.randn(shape, dtype=torch.float32, device="cuda")
    gamma = torch.randn(C, dtype=torch.float32, device="cuda")
    beta = torch.randn(C, dtype=torch.float32, device="cuda")
    golden = F.group_norm(input_t, G, gamma, beta, eps)

    input_cuda = input_t
    actual = cuda_op.group_norm(input_cuda, gamma, beta, G, eps)
    torch.testing.assert_close(golden, actual, atol=1.3e-6, rtol=1e-5)
