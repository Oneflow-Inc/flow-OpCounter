import oneflow as flow
import oneflow.nn as nn
import sys
sys.path.append("./")
from flowflops import get_model_complexity_info


input_size = 160
hidden_size = 512

models = {
    "RNNCell": nn.Sequential(nn.RNNCell(input_size, hidden_size)),
    "GRUCell": nn.Sequential(nn.GRUCell(input_size, hidden_size)),
    "LSTMCell": nn.Sequential(nn.LSTMCell(input_size, hidden_size)),
    "RNN": nn.Sequential(nn.RNN(input_size, hidden_size)),
    "GRU": nn.Sequential(nn.GRU(input_size, hidden_size)),
    "LSTM": nn.Sequential(nn.LSTM(input_size, hidden_size)),
    "stacked-RNN": nn.Sequential(nn.RNN(input_size, hidden_size, num_layers=4)),
    "stacked-GRU": nn.Sequential(nn.GRU(input_size, hidden_size, num_layers=4)),
    "stacked-LSTM": nn.Sequential(nn.LSTM(input_size, hidden_size, num_layers=4)),
    "BiRNN": nn.Sequential(nn.RNN(input_size, hidden_size, bidirectional=True)),
    "BiGRU": nn.Sequential(nn.GRU(input_size, hidden_size, bidirectional=True)),
    "BiLSTM": nn.Sequential(nn.LSTM(input_size, hidden_size, bidirectional=True)),
    "stacked-BiRNN": nn.Sequential(
        nn.RNN(input_size, hidden_size, bidirectional=True, num_layers=4)
    ),
    "stacked-BiGRU": nn.Sequential(
        nn.GRU(input_size, hidden_size, bidirectional=True, num_layers=4)
    ),
    "stacked-BiLSTM": nn.Sequential(
        nn.LSTM(input_size, hidden_size, bidirectional=True, num_layers=4)
    ),
}

print("{} | {} | {}".format("Model", "Params(M)", "FLOPs(G)"))
print("---|---|---")

for name, model in models.items():
    # time_first dummy inputs
    inputs = flow.randn(100, 32, input_size)
    if name.find("Cell") != -1:
        total_ops, total_params = get_model_complexity_info(
            model, tuple(inputs[0].shape),
            as_strings=False,
            print_per_layer_stat=False
        )
    else:
        total_ops, total_params = get_model_complexity_info(
            model, tuple(inputs.shape),
            as_strings=False,
            print_per_layer_stat=False
        )
    print(
        "{} | {:.2f} | {:.2f}".format(
            name,
            total_params,
            total_ops,
        )
    )

# validate batch_first support
inputs = flow.randn(1, 32, input_size)
ops_time_first, params_time_first = get_model_complexity_info(
    nn.Sequential(nn.LSTM(input_size, hidden_size)), 
    tuple(inputs.shape), 
    as_strings=False,
    print_per_layer_stat=False
)
ops_batch_first, params_batch_first = get_model_complexity_info(
    nn.Sequential(nn.LSTM(input_size, hidden_size, batch_first=True)), 
    tuple(inputs.transpose(0, 1).shape),
    as_strings=False,
    print_per_layer_stat=False
)
assert ops_batch_first == ops_time_first
assert params_batch_first == params_time_first
