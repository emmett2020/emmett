import torch
import cuda_op

def test_basic():
    a = torch.randn(1,2,3, dtype=torch.float32)
    golden = torch.sigmoid(a)
    actual = cuda_op.sigmoid(a.to("cuda")).to("cpu")
    torch.testing.assert_close(golden, actual, atol=1e-5, rtol=1e-5)

