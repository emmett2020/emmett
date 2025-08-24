import torch
import cuda_op


def test_perf():
    N = 16
    C = 64
    H = 256
    W = 256
    eps = 1e-5
    shape = [N, C, H, W]
    normalized_shape = (C, H, W)

    therotical_time = (20 * N * C * H * W) / (504 * 10**3)
    print(f"\ntherotical performance: {therotical_time}us")

    x = torch.randn(shape, device='cuda')
    gamma = torch.ones(normalized_shape, device='cuda')

    golden = torch.nn.functional.rms_norm(x,
                                          normalized_shape,
                                          weight=gamma,
                                          eps=eps)
    actual = cuda_op.rms_norm(x, eps)

    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1.3e-6)
