import torch
import cuda_op
import torch.nn as nn
import torch.nn.functional as F


def test_basic():
    N, C, H, W = 2, 4, 8, 8
    epsilon = 1e-5
    momentum = 0.9

    input_tensor = torch.randn(N, C, H, W)
    gamma = torch.randn(C)
    beta = torch.randn(C)
    running_mean = torch.zeros(C)
    running_var = torch.ones(C)

    pt_input = input_tensor.clone().detach().requires_grad_(False)
    pt_gamma = gamma.clone().detach()
    pt_beta = beta.clone().detach()
    pt_running_mean = running_mean.clone().detach()
    pt_running_var = running_var.clone().detach()

    print("=== training mode ===")
    mean = pt_input.mean(dim=(0, 2, 3))
    var = pt_input.var(dim=(0, 2, 3), unbiased=False)
    golden = F.batch_norm(pt_input,
                          mean,
                          var,
                          pt_gamma,
                          pt_beta,
                          training=True,
                          eps=epsilon)

    actual = torch.empty_like(input_tensor, device="cuda")
    cuda_op.batch_norm(pt_input, mean, var, pt_gamma, pt_beta, epsilon,
                       momentum, True)
    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1e-5)
