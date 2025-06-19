import torch
import cuda_op

shape = [1, 2]


def test_basic():
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
