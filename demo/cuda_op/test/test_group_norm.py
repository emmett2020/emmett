import torch
import cuda_op
import torch.nn as nn
import torch.nn.functional as F


def test_basic():
    eps = 1e-5

    N = 1
    C = 16
    H = 4
    W = 16
    G = 4
    shape = [N, C, H, W]

    therotical_time = (12 * N * C * H * W) / (504 * 10**3)
    print(f"\ntherotical performance: {therotical_time}us")

    input_t = torch.randn(shape, dtype=torch.float32)
    gamma = torch.ones(C, dtype=torch.float32)
    beta = torch.zeros(C, dtype=torch.float32)
    golden = F.group_norm(input_t, G, gamma, beta, eps)

    for group_idx in range(int(C / G)):
        mean = input_t[0, group_idx * G:(group_idx + 1) * G:].mean().item()
        var = input_t[0, group_idx * G:(group_idx + 1) * G:].var().item()
        std = input_t[0, group_idx * G:(group_idx + 1) * G:].std().item()
        rstd = 1 / input_t[0, group_idx * G:(group_idx + 1) * G:].std().item()

        print(
            f"group_idx: {group_idx}, mean: {mean:.6f}, var: {var:.6f} std: {std:.6f} rstd:{rstd:.6f}"
        )

    input_cuda = input_t.to("cuda")
    actual = cuda_op.group_norm(input_cuda, gamma.to("cuda"), beta.to("cuda"),
                                G, eps).to("cpu")
    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1e-5)
