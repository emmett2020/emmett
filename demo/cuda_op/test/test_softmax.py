import torch
import torch.nn.functional as F
from py_impl.softmax import base_softmax, safe_softmax
import cuda_op


def test_softmax_dim():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 4)
    print(x)

    for dim in range(3):
        print("dim=", dim)
        o = F.softmax(x, dim)
        print(o.shape)
        print(o)


def test_softmax_python_impl():
    x = torch.randn(2, 2, 3, dtype=torch.float)
    dim = 0

    o_golden = torch.softmax(x, dim)
    o_base = base_softmax(x, dim)
    o_safe = safe_softmax(x, dim)

    torch.testing.assert_close(o_golden, o_base, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(o_golden, o_safe, atol=1e-5, rtol=1e-5)
