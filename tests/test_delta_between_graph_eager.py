import pytest
import unittest
import oneflow as flow
import oneflow.nn as nn
import flowvision

import sys
sys.path.append("./")
from flowflops import get_model_complexity_info


class TestNet(nn.Module):
    def __init__(self, num_classes: int = 1000) -> None:
        super(TestNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2, bias=False),
            nn.BatchNorm2d(64, 64),
            nn.BatchNorm2d(64, 64, track_running_stats=False),
            nn.ELU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.InstanceNorm2d(64, 64),
            nn.PReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            # nn.GroupNorm(1, 384),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU6(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096, bias=False),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes, bias=True),
        )

    def forward(self, x: flow.Tensor) -> flow.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = flow.flatten(x, 1)
        x = self.classifier(x)
        return x


class TestLayers(unittest.TestCase):
    def test_conv(self):
        net = TestNet()
        flops_in_different_modes = dict()
        for mode in ["eager", "graph"]:
            flops, _ = get_model_complexity_info(
                net, (4, 3, 227, 227),
                as_strings=False,
                print_per_layer_stat=False,
                mode=mode
            )
            flops_in_different_modes[mode] = int(flops)
        assert flops_in_different_modes["eager"] == flops_in_different_modes["graph"]
    
    def test_resnet(self):
        net = flowvision.models.resnet18()
        flops_in_different_modes = dict()
        for mode in ["eager", "graph"]:
            flops, _ = get_model_complexity_info(
                net, (4, 3, 224, 224),
                as_strings=False,
                print_per_layer_stat=False,
                mode=mode
            )
            flops_in_different_modes[mode] = int(flops)
        print(flops_in_different_modes)
        assert int(flops_in_different_modes["graph"]) - int(flops_in_different_modes["eager"]) == 2 * (
            256 * 56 * 56 + \
            512 * 28 * 28 + \
            1024 * 14 * 14 + \
            2048 * 7 * 7
        )



if __name__ == "__main__":
    unittest.main()
