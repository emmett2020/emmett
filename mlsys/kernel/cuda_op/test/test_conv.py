import torch
import torch.nn.functional as F
import cuda_op
import torch.nn.functional as F


def test_perf():
    """Test sort"""
    IH, IW, KH, KW = 1024, 1024, 7, 7

    # (batch_size, channels, height, width)
    x = torch.randn(1, 1, IH, IW, dtype=torch.float, device="cuda")

    # (out_channels, in_channels/groups, kernel_height, kernel_width)
    kernel = torch.randn(1, 1, KH, KW, dtype=torch.float, device="cuda")

    golden = F.conv2d(input=x,
                      weight=kernel,
                      bias=None,
                      stride=1,
                      padding=0,
                      dilation=1,
                      groups=1)
    actual = cuda_op.conv2d(x, kernel)

    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1.3e-6)
