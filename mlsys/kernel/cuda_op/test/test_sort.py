import torch
import torch.nn.functional as F
import cuda_op


def test_perf():
    """Test sort"""
    N = 1024 * 1024
    x = torch.randn(N, dtype=torch.float, device="cuda")

    golden = torch.sort(x)[0]
    actual = cuda_op.sort(x)
    torch.testing.assert_close(golden, actual, atol=0, rtol=0)
