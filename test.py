import oneflow as flow


flow.manual_seed(987342)
for i in range(5):
    n, in_c, out_c = flow.randint(1, 500, (3,)).tolist()
    ops = n * in_c * out_c
    print(ops)
    print(flow.DoubleTensor([ops]))
