import torch
import cuda_op


def test_basic():
    shape = [1, 2]
    x = torch.randn(shape, device='cpu')
    gamma = torch.ones(shape[1], device='cpu')
    beta = torch.zeros(shape[1], device='cpu')
    golden = torch.nn.functional.layer_norm(x, [shape[1]],
                                            weight=gamma,
                                            bias=beta)
    x = x.cuda()
    gamma = gamma.cuda()
    beta = beta.cuda()
    actual = cuda_op.layer_norm(x, gamma, beta, 1e-5)
    torch.testing.assert_close(golden.cpu(),
                               actual.cpu(),
                               atol=1e-5,
                               rtol=1e-5)


def test_perf():
    N = 16
    C = 128
    H = 256
    W = 256
    eps = 1e-5
    shape = [N, C, H, W]
    normalized_shape = (C, H, W)

    x = torch.randn(shape, device='cuda')
    gamma = torch.randn(normalized_shape, device='cuda')
    beta = torch.randn(normalized_shape, device='cuda')
    golden = torch.nn.functional.layer_norm(x,
                                            normalized_shape,
                                            weight=gamma,
                                            bias=beta,
                                            eps=eps)
    # actual = cuda_op.layer_norm(x, gamma, beta, eps)
    # torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1.3e-6)
