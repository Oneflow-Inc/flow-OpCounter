# OneFlow-OpCounter | [**简体中文**](README_CN.md)

[![PyPI version](https://img.shields.io/pypi/v/flowflops.svg)](https://pypi.org/project/flowflops/)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/flowflops.svg)](https://pypi.org/project/flowflops/)
[![PyPI license](https://img.shields.io/pypi/l/flowflops.svg)](https://pypi.org/project/flowflops/)

modified from `https://github.com/sovrasov/flops-counter.pytorch`

## install

```shell
pip install flowflops
```

## usage

```python
import oneflow as flow
from flowflops import get_model_complexity_info
from flowflops.utils import flops_to_string, params_to_string


model = ...              # your own model, nn.Module
dsize = (1, 3, 224, 224) # B, C, H, W

total_flops, total_params = get_model_complexity_info(
    model, dsize,
    as_strings=False,
    print_per_layer_stat=False,
    mode="eager"         # eager or graph
)
print(flops_to_string(total_flops), params_to_string(total_params))
```

## why graph?

```python
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        # !!!NOTE!!!: this will make add-op flops that cannot be hooked in eager mode
        out = self.relu(out)

        return out
```

## sample

`python benchmark/evaluate_famous_models.py`

```
====== eager ======
+--------------------+----------+-------------+
|       Model        |  Params  |    FLOPs    |
+--------------------+----------+-------------+
|      alexnet       |  61.1 M  | 718.16 MMac |
|       vgg11        | 132.86 M |  7.63 GMac  |
|      vgg11_bn      | 132.87 M |  7.64 GMac  |
|   squeezenet1_0    |  1.25 M  | 830.05 MMac |
|   squeezenet1_1    |  1.24 M  | 355.86 MMac |
|      resnet18      | 11.69 M  |  1.82 GMac  |
|      resnet50      | 25.56 M  |  4.12 GMac  |
|  resnext50_32x4d   | 25.03 M  |  4.27 GMac  |
| shufflenet_v2_x0_5 |  1.37 M  |  43.65 MMac |
|   regnet_x_16gf    | 54.28 M  |  16.01 GMac |
|  efficientnet_b0   |  5.29 M  | 401.67 MMac |
|    densenet121     |  7.98 M  |  2.88 GMac  |
+--------------------+----------+-------------+
====== graph ======
+--------------------+----------+-------------+
|       Model        |  Params  |    FLOPs    |
+--------------------+----------+-------------+
|      alexnet       |  61.1 M  | 718.16 MMac |
|       vgg11        | 132.86 M |  7.63 GMac  |
|      vgg11_bn      | 132.87 M |  7.64 GMac  |
|   squeezenet1_0    |  1.25 M  | 830.05 MMac |
|   squeezenet1_1    |  1.24 M  | 355.86 MMac |
|      resnet18      | 11.69 M  |  1.82 GMac  |
|      resnet50      | 25.56 M  |  4.13 GMac  |
|  resnext50_32x4d   | 25.03 M  |  4.28 GMac  |
| shufflenet_v2_x0_5 |  1.37 M  |  43.7 MMac  |
|   regnet_x_16gf    | 54.28 M  |  16.02 GMac |
|  efficientnet_b0   |  5.29 M  | 410.35 MMac |
|    densenet121     |  7.98 M  |  2.88 GMac  |
+--------------------+----------+-------------+
```

## support

### Eager

> the outputs will be the same as the `ptflops`

supported layers:

```python
# convolutions
nn.Conv1d
nn.Conv2d
nn.Conv3d
# activations
nn.ReLU
nn.PReLU
nn.ELU
nn.LeakyReLU
nn.ReLU6
# poolings
nn.MaxPool1d
nn.AvgPool1d
nn.AvgPool2d
nn.MaxPool2d
nn.MaxPool3d
nn.AvgPool3d
# nn.AdaptiveMaxPool1d
nn.AdaptiveAvgPool1d
# nn.AdaptiveMaxPool2d
nn.AdaptiveAvgPool2d
# nn.AdaptiveMaxPool3d
nn.AdaptiveAvgPool3d
# BNs
nn.BatchNorm1d
nn.BatchNorm2d
nn.BatchNorm3d
# INs
nn.InstanceNorm1d
nn.InstanceNorm2d
nn.InstanceNorm3d
# FC
nn.Linear
# Upscale
nn.Upsample
# Deconvolution
nn.ConvTranspose1d
nn.ConvTranspose2d
nn.ConvTranspose3d
# RNN
nn.RNN
nn.GRU
nn.LSTM
nn.RNNCell
nn.LSTMCell
nn.GRUCell
```

### Graph

supported ops:

```python
# conv
"conv1d"
"conv2d"
"conv3d"
# pool
"max_pool_1d"
"max_pool_2d"
"max_pool_3d"
"avg_pool_1d"
"avg_pool_2d"
"avg_pool_3d"
"adaptive_max_pool1d"
"adaptive_max_pool2d"
"adaptive_max_pool3d"
"adaptive_avg_pool1d"
"adaptive_avg_pool2d"
"adaptive_avg_pool3d"
# activate
"relu"
"leaky_relu"
"prelu"
"hardtanh"
"elu"
"silu"
"sigmoid"
"sigmoid_v2"
# add
"bias_add"
"add_n"
# matmul
"matmul"
"broadcast_matmul"
# norm
"normalization"
# scalar
"scalar_mul"
"scalar_add"
"scalar_sub"
"scalar_div"
# stats
"var"
# math
"sqrt"
"reduce_sum"
# broadcast
"broadcast_mul"
"broadcast_add"
"broadcast_sub"
"broadcast_div"
# empty
"reshape"
"ones_like"
"zero_like"
"flatten"
"concat"
"transpose"
"slice"
```
