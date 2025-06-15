import torch
import cuda_op
import torch.nn as nn
import torch.nn.functional as F


def test_basic():
    num_groups = 4
    num_channels = 16
    eps = 1e-5
    shape = [1, num_channels, 4, 16]

    input_t = torch.randn(shape, dtype=torch.float32)
    gamma = torch.ones(num_channels, dtype=torch.float32)
    beta = torch.zeros(num_channels, dtype=torch.float32)
    golden = F.group_norm(input_t, num_groups, gamma, beta, eps)

    for group_idx in range(int(num_channels / num_groups)):
        mean = input_t[0, group_idx * num_groups:(group_idx + 1) *
                       num_groups:].mean().item()
        var = input_t[0, group_idx * num_groups:(group_idx + 1) *
                      num_groups:].var().item()
        std = input_t[0, group_idx * num_groups:(group_idx + 1) *
                      num_groups:].std().item()
        rstd = 1 / input_t[0, group_idx * num_groups:(group_idx + 1) *
                           num_groups:].std().item()

        print(
            f"group_idx: {group_idx}, mean: {mean:.6f}, var: {var:.6f} std: {std:.6f} rstd:{rstd:.6f}"
        )

    input_cuda = input_t.to("cuda")
    actual = cuda_op.group_norm(input_cuda, gamma.to("cuda"), beta.to("cuda"),
                                num_groups, eps).to("cpu")
    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1e-5)
