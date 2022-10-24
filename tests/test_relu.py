import pytest
import oneflow as flow
import oneflow.nn as nn
from thop import profile


class TestUtils:
    def test_relu(self):
        n, in_c, out_c = 1, 100, 200
        data = flow.randn(n, in_c)
        net = nn.ReLU()
        flops, params = profile(net, inputs=(flow.randn(n, in_c), ))
        print(flops, params)
        assert flops == 0
