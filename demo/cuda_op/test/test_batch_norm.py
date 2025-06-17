import torch
import cuda_op
import torch.nn as nn
import torch.nn.functional as F


def test_training():
    N, C, H, W = 2, 4, 8, 8
    epsilon = 1e-5
    momentum = 0.9

    input = torch.randn(N, C, H, W)
    gamma = torch.randn(C)
    beta = torch.randn(C)
    running_mean = torch.zeros(C)
    running_var = torch.ones(C)
    cuda_running_mean = running_mean.to("cuda").clone().detach()
    cuda_running_var = running_var.to("cuda").clone().detach()

    print("=== training mode ===")
    mean = input.mean(dim=(0, 2, 3))
    var = input.var(dim=(0, 2, 3), unbiased=False)
    golden = F.batch_norm(input, running_mean, running_var, gamma, beta,
                          training=True,
                          eps=epsilon)

    input = input.to("cuda")
    gamma = gamma.to("cuda")
    beta = beta.to("cuda")
    running_mean = cuda_running_mean
    running_var = cuda_running_var
    actual = cuda_op.batch_norm(input, running_mean, running_var, gamma, beta, epsilon,
                                momentum, True)
    actual = actual.to("cpu")
    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1e-5)
