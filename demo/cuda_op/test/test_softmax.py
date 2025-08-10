import torch
import torch.nn.functional as F
from py_impl.softmax import base_softmax, safe_softmax, online_softmax, block_online_softmax
import cuda_op


def test_softmax_dim():
    """test softmax dim parameter"""
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    print(x)

    for dim in range(3):
        print("dim=", dim)
        o = F.softmax(x, dim)
        print(o.shape)
        print(o)


def test_softmax_python_impl():
    """test softmax python implementation"""
    x = torch.randn(2, 2, 3, dtype=torch.float)
    dim = 0

    o_golden = torch.softmax(x, dim)
    o_base = base_softmax(x, dim)
    o_safe = safe_softmax(x, dim)
    o_online = online_softmax(x, dim)
    o_block_online = block_online_softmax(x, dim)

    torch.testing.assert_close(o_golden, o_base, atol=1e-5, rtol=1.3e-6)
    torch.testing.assert_close(o_golden, o_safe, atol=1e-5, rtol=1.3e-6)
    torch.testing.assert_close(o_golden, o_online, atol=1e-5, rtol=1.3e-6)
    torch.testing.assert_close(o_golden,
                               o_block_online,
                               atol=1e-5,
                               rtol=1.3e-6)


def test_softmax_perf():
    """Test cuda op softmax nhwc version"""
    N, H, W, C = 16, 256, 256, 128
    x = torch.randn(N, H, W, C, dtype=torch.float, device="cuda")

    golden = torch.softmax(x, -1)
    actual = cuda_op.softmax(x)

    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1.3e-6)


def test_safe_softmax_perf():
    """Test cuda op softmax nhwc version"""
    N, H, W, C = 16, 256, 256, 128
    x = torch.randn(N, H, W, C, dtype=torch.float, device="cuda")

    golden = torch.softmax(x, -1)
    actual = cuda_op.safe_softmax(x)

    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1.3e-6)
