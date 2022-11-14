import pytest
import unittest
import oneflow
import oneflow.nn as nn

import sys
sys.path.append("./")
from flowflops import get_model_complexity_info


class TestLayers(unittest.TestCase):
    def test_conv(self):
        net = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, bias=False))
        for mode in ["eager", "graph"]:
            flops, params = get_model_complexity_info(
                net, (3, 64, 224, 224),
                as_strings=False,
                print_per_layer_stat=False,
                mode=mode
            )
            assert params == 3 * 3 * 64 * 128
            assert int(flops) == 3 * 3 * 224 * 224 * 64 * 128

    def test_group_conv(self):
        net = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1, groups=64, bias=False))
        for mode in ["eager", "graph"]:
            flops, params = get_model_complexity_info(
                net, (2, 64, 224, 224),
                as_strings=False,
                print_per_layer_stat=False,
                mode=mode
            )
            assert params == 3 * 3 * 64 * 128 // 64
            assert int(flops) == 3 * 3 * 224 * 224 * 128 * 1

    def test_linear_case1(self):
        n, in_c, out_c = 5, 100, 200
        net = nn.Linear(in_c, out_c, bias=False)
        for mode in ["eager", "graph"]:
            flops, params = get_model_complexity_info(
                net, (n, 15, in_c),
                as_strings=False,
                print_per_layer_stat=False,
                mode=mode
            )
            assert params == in_c * out_c
            assert int(flops) == int(n * in_c * out_c * 15)

    def test_linear_case2(self):
        for i in range(10):
            n, in_c, out_c = oneflow.randint(1, 500, (3,)).tolist()
            net = nn.Linear(in_c, out_c)
            for mode in ["eager", "graph"]:
                flops, params = get_model_complexity_info(
                    net, (n, in_c),
                    as_strings=False,
                    print_per_layer_stat=False,
                    mode=mode
                )
                assert params == in_c * out_c + out_c
                assert int(flops) == n * (in_c * out_c) + out_c

    def test_relu(self):
        n, in_c = 1, 100
        net = nn.ReLU(inplace=True)
        for mode in ["eager", "graph"]:
            flops, params = get_model_complexity_info(
                net, (n, in_c),
                as_strings=False,
                print_per_layer_stat=False,
                mode=mode
            )
            assert int(flops) == n * in_c
            assert params == 0

    def test_pool(self):
        n, h, w = 1, 64, 64
        net = nn.MaxPool2d(2, 2)
        for mode in ["eager", "graph"]:
            flops, params = get_model_complexity_info(
                net, (n, 1, h, w),
                as_strings=False,
                print_per_layer_stat=False
            )
            assert int(flops) == 2 * 2 * (h // 2) * (w // 2)
            assert params == 0


if __name__ == "__main__":
    unittest.main()
