import pytest
import oneflow as flow
import oneflow.nn as nn
from thop import profile


class TestUtils:
    def test_matmul_case2(self):
        n, in_c, out_c = 1, 100, 200
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(flow.randn(n, in_c), ))
        # print(flops, params)
        assert flops == n * in_c * out_c
        assert params != 0

    def test_matmul_case2(self):
        for i in range(20):
            n, in_c, out_c = flow.randint(1, 500, (3,)).tolist()
            net = nn.Linear(in_c, out_c)
            flops, params = profile(net, inputs=(flow.randn(n, in_c), ))
            print(flops, params)
            assert flops == n * in_c * out_c
            assert params != 0

    def test_conv2d(self):
        n, in_c, out_c = flow.randint(1, 500, (3,)).tolist()
        net = nn.Linear(in_c, out_c)
        flops, params = profile(net, inputs=(flow.randn(n, in_c), ))
        # print(flops, params)
        assert flops == n * in_c * out_c
        assert params != 0
