import oneflow as flow
from flowvision import models
from prettytable import PrettyTable

import sys
sys.path.append("./")
from flowflops import get_model_complexity_info
from flowflops.utils import flops_to_string, params_to_string


def main(mode, model_names):
    device = "cpu"
    if flow.cuda.is_available():
        device = "cuda"
    print("====== {} ======".format(mode))

    table = PrettyTable(['Model', 'Params', 'FLOPs'])

    for name in model_names:
        try:
            model = models.__dict__[name]().to(device)
        except:
            continue
        dsize = (1, 3, 224, 224)
        if "alexnet" in name:
            dsize = (1, 3, 227, 227)
        if "inception" in name:
            dsize = (1, 3, 299, 299)
        total_flops, total_params = get_model_complexity_info(
            model, dsize,
            as_strings=False,
            print_per_layer_stat=False,
            mode=mode
        )
        table.add_row([name, params_to_string(total_params), flops_to_string(int(total_flops))])
    print(table)


if __name__ == "__main__":
    model_names = [
        "alexnet",
        "vgg11",
        "vgg11_bn",
        "squeezenet1_0",
        "squeezenet1_1",
        "resnet18",
        "resnet50",
        "resnext50_32x4d",
        "shufflenet_v2_x0_5",
        "regnet_x_16gf",
        "efficientnet_b0",
        "densenet121",
    ]
    for mode in ["eager", "graph"]:
        main(mode, model_names)
