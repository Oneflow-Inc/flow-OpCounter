import oneflow as flow
from flowvision import models
import sys
sys.path.join("./")
from flowflops import get_model_complexity_info


model_names = sorted(
    name
    for name in models.__dict__
    if name.islower()
    and not name.startswith("__")  # and "inception" in name
    and callable(models.__dict__[name])
)

print("%s | %s | %s" % ("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")

device = "cpu"
if flow.cuda.is_available():
    device = "cuda"

for name in model_names:
    model = models.__dict__[name]().to(device)
    dsize = (1, 3, 224, 224)
    if "inception" in name:
        dsize = (1, 3, 299, 299)
    total_ops, total_params = flops, params = get_model_complexity_info(
        net, dsize,
        as_strings=False,
        print_per_layer_stat=False
    )
    print(
        "%s | %.2f | %.2f" % (name, total_params, total_ops)
    )
