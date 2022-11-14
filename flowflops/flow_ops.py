"""
Copyright (C) 2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
"""

import math
import numpy as np
import oneflow.nn as nn


# --------------------------------
# For Graph
# --------------------------------
def empty_flops_counter(module, input, output):
    return 0


def conv_flops_counter(attr, input_strs, op_name2op_shape):
    strides = attr["strides"].at_list_int32.val
    padding = attr["padding_before"].at_list_int32.val
    kernel_size = attr["kernel_size"].at_list_int32.val
    groups = attr["groups"].at_int32

    input_shape = op_name2op_shape[input_strs["in"].s[0]]
    kernel_shape = op_name2op_shape[input_strs["weight"].s[0]]

    in_dims = input_shape[2:]
    output_dims = [
        math.ceil((in_dims[0] - kernel_size[0] + 2 * padding[0]) / strides[0]) + 1,
        math.ceil((in_dims[1] - kernel_size[1] + 2 * padding[1]) / strides[1]) + 1
    ]
    in_channels = input_shape[1]
    out_channels = kernel_shape[0]

    conv_per_position_flops = int(np.prod(kernel_size) * in_channels * out_channels / groups)
    active_elements_count = int(np.prod(output_dims))

    return int(conv_per_position_flops * active_elements_count)


def pool_flops_counter(attr, input_strs, op_name2op_shape):
    # also broadcast and activate
    input_shape = op_name2op_shape[input_strs["x"].s[0]]
    return int(np.prod(input_shape[1:]))


def matmul_flops_counter(attr, input_strs, op_name2op_shape):
    a_shape = op_name2op_shape[input_strs["a"].s[0]]
    b_shape = op_name2op_shape[input_strs["b"].s[0]]
    return int((np.prod(a_shape) * np.prod(b_shape[:-1])))


def add_n_flops_counter(attr, input_strs, op_name2op_shape):
    in_shapes = []
    for v in input_strs["in"].s:
        in_shapes.append(op_name2op_shape[v])
    return int(np.prod(in_shapes[0]) * (len(in_shapes) - 1))


def bias_add_flops_counter(attr, input_strs, op_name2op_shape):
    input_shape = op_name2op_shape[input_strs["a"].s[0]]
    return int(np.prod(input_shape[1:]))


def broadcast_flops_counter(attr, input_strs, op_name2op_shape):
    x_shape = op_name2op_shape[input_strs["x"].s[0]]
    y_shape = op_name2op_shape[input_strs["y"].s[0]]
    return int(np.prod(y_shape))


def normalization_flops_counter(attr, input_strs, op_name2op_shape):
    flops: int = 0
    input_shape = op_name2op_shape[input_strs["x"].s[0]]
    flops += int(np.prod(input_shape[1:])) * 2
    if hasattr(input_strs, "moving_mean") and hasattr(input_strs, "moving_variance"):
        flops += int(np.prod(input_shape[1:])) * 2
    return int(flops)


def scalar_flops_counter(attr, input_strs, op_name2op_shape):
    input_shape = op_name2op_shape[input_strs["in"].s[0]]
    return int(np.prod(input_shape[1:]))


def reduce_flops_counter(attr, input_strs, op_name2op_shape):
    input_shape = op_name2op_shape[input_strs["input_tensor"].s[0]]
    return int(np.prod(input_shape[1:]))


# --------------------------------
# For Eager
# --------------------------------
def empty_flops_counter_hook(module, input, output):
    module.__flops__ += 0


def upsample_flops_counter_hook(module, input, output):
    output_size = output[0]
    batch_size = output_size.shape[0]
    output_elements_count = batch_size
    for val in output_size.shape[1:]:
        output_elements_count *= val
    module.__flops__ += int(output_elements_count)


def relu_flops_counter_hook(module, input, output):
    active_elements_count = output.numel()
    module.__flops__ += int(active_elements_count)


def linear_flops_counter_hook(module, input, output):
    input = input[0]
    batch_size = input.shape[0]
    output_last_dim = output.shape[-1]
    bias_flops = output_last_dim if module.bias is not None else 0
    module.__flops__ += int((np.prod(input.shape) * output_last_dim + bias_flops) * batch_size)


def pool_flops_counter_hook(module, input, output):
    input = input[0]
    module.__flops__ += int(np.prod(input.shape))


def bn_flops_counter_hook(module, input, output):
    input = input[0]
    batch_flops = np.prod(input.shape)
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def in_flops_counter_hook(module, input, output):
    input = input[0]
    # NOTE(hujiakui): IN splited in oneflow, 5 ops added
    batch_flops = np.prod(input.shape) + 10 * np.prod(input.shape[:2])
    if module.affine:
        batch_flops *= 2
    module.__flops__ += int(batch_flops)


def conv_flops_counter_hook(conv_module, input, output):
    # Can have multiple inputs, getting the first one
    input = input[0]

    batch_size = input.shape[0]
    output_dims = list(output.shape[2:])

    kernel_dims = list(conv_module.kernel_size)
    in_channels = conv_module.in_channels
    out_channels = conv_module.out_channels
    groups = conv_module.groups

    filters_per_channel = out_channels // groups
    conv_per_position_flops = int(np.prod(kernel_dims)) * \
        in_channels * filters_per_channel

    active_elements_count = batch_size * int(np.prod(output_dims))

    overall_conv_flops = conv_per_position_flops * active_elements_count

    bias_flops = 0
    if conv_module.bias is not None:
        bias_flops = out_channels * active_elements_count

    overall_flops = overall_conv_flops + bias_flops

    conv_module.__flops__ += int(overall_flops)


def rnn_flops(flops, rnn_module, w_ih, w_hh, input_size):
    # matrix matrix mult ih state and internal state
    flops += w_ih.shape[0]*w_ih.shape[1]
    # matrix matrix mult hh state and internal state
    flops += w_hh.shape[0]*w_hh.shape[1]
    if isinstance(rnn_module, (nn.RNN, nn.RNNCell)):
        # add both operations
        flops += rnn_module.hidden_size
    elif isinstance(rnn_module, (nn.GRU, nn.GRUCell)):
        # hadamard of r
        flops += rnn_module.hidden_size
        # adding operations from both states
        flops += rnn_module.hidden_size*3
        # last two hadamard product and add
        flops += rnn_module.hidden_size*3
    elif isinstance(rnn_module, (nn.LSTM, nn.LSTMCell)):
        # adding operations from both states
        flops += rnn_module.hidden_size*4
        # two hadamard product and add for C state
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
        # final hadamard
        flops += rnn_module.hidden_size + rnn_module.hidden_size + rnn_module.hidden_size
    return flops


def rnn_flops_counter_hook(rnn_module, input, output):
    """
    Takes into account batch goes at first position, contrary
    to oneflow common rule (but actually it doesn't matter).
    If sigmoid and tanh are hard, only a comparison FLOPS should be accurate
    """
    flops = 0
    # input is a tuple containing a sequence to process and (optionally) hidden state
    input = input[0]
    batch_size = input.shape[0]
    seq_length = input.shape[1]
    num_layers = rnn_module.num_layers

    for i in range(num_layers):
        w_ih = rnn_module.__getattr__('weight_ih_l' + str(i))
        w_hh = rnn_module.__getattr__('weight_hh_l' + str(i))
        if i == 0:
            input_size = rnn_module.input_size
        else:
            input_size = rnn_module.hidden_size
        flops = rnn_flops(flops, rnn_module, w_ih, w_hh, input_size)
        if rnn_module.bias:
            b_ih = rnn_module.__getattr__('bias_ih_l' + str(i))
            b_hh = rnn_module.__getattr__('bias_hh_l' + str(i))
            flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    flops *= seq_length
    if rnn_module.bidirectional:
        flops *= 2
    rnn_module.__flops__ += int(flops)


def rnn_cell_flops_counter_hook(rnn_cell_module, input, output):
    flops = 0
    input = input[0]
    batch_size = input.shape[0]
    w_ih = rnn_cell_module.__getattr__('weight_ih')
    w_hh = rnn_cell_module.__getattr__('weight_hh')
    input_size = input.shape[1]
    flops = rnn_flops(flops, rnn_cell_module, w_ih, w_hh, input_size)
    if rnn_cell_module.bias:
        b_ih = rnn_cell_module.__getattr__('bias_ih')
        b_hh = rnn_cell_module.__getattr__('bias_hh')
        flops += b_ih.shape[0] + b_hh.shape[0]

    flops *= batch_size
    rnn_cell_module.__flops__ += int(flops)


def multihead_attention_counter_hook(multihead_attention_module, input, output):
    flops = 0

    q, k, v = input

    batch_first = multihead_attention_module.batch_first \
        if hasattr(multihead_attention_module, 'batch_first') else False
    if batch_first:
        batch_size = q.shape[0]
        len_idx = 1
    else:
        batch_size = q.shape[1]
        len_idx = 0

    dim_idx = 2

    qdim = q.shape[dim_idx]
    kdim = k.shape[dim_idx]
    vdim = v.shape[dim_idx]

    qlen = q.shape[len_idx]
    klen = k.shape[len_idx]
    vlen = v.shape[len_idx]

    num_heads = multihead_attention_module.num_heads
    assert qdim == multihead_attention_module.embed_dim

    if multihead_attention_module.kdim is None:
        assert kdim == qdim
    if multihead_attention_module.vdim is None:
        assert vdim == qdim

    flops = 0

    # Q scaling
    flops += qlen * qdim

    # Initial projections
    flops += (
        (qlen * qdim * qdim)  # QW
        + (klen * kdim * kdim)  # KW
        + (vlen * vdim * vdim)  # VW
    )

    if multihead_attention_module.in_proj_bias is not None:
        flops += (qlen + klen + vlen) * qdim

    # attention heads: scale, matmul, softmax, matmul
    qk_head_dim = qdim // num_heads
    v_head_dim = vdim // num_heads

    head_flops = (
        (qlen * klen * qk_head_dim)  # QK^T
        + (qlen * klen)  # softmax
        + (qlen * klen * v_head_dim)  # AV
    )

    flops += num_heads * head_flops

    # final projection, bias is always enabled
    flops += qlen * vdim * (vdim + 1)

    flops *= batch_size
    multihead_attention_module.__flops__ += int(flops)


CUSTOM_MODULES_MAPPING = {}

GRAPH_FLOPS_COUNT_FUNC = {
    # conv
    "conv1d": conv_flops_counter,
    "conv2d": conv_flops_counter,
    "conv3d": conv_flops_counter,
    # pool
    "max_pool_1d": pool_flops_counter,
    "max_pool_2d": pool_flops_counter,
    "max_pool_3d": pool_flops_counter,
    "avg_pool_1d": pool_flops_counter,
    "avg_pool_2d": pool_flops_counter,
    "avg_pool_3d": pool_flops_counter,
    "adaptive_max_pool1d": pool_flops_counter,
    "adaptive_max_pool2d": pool_flops_counter,
    "adaptive_max_pool3d": pool_flops_counter,
    "adaptive_avg_pool1d": pool_flops_counter,
    "adaptive_avg_pool2d": pool_flops_counter,
    "adaptive_avg_pool3d": pool_flops_counter,
    # activate
    "relu": pool_flops_counter,
    "leaky_relu": pool_flops_counter,
    "prelu": pool_flops_counter,
    "hardtanh": scalar_flops_counter,
    "elu": scalar_flops_counter,
    "silu": scalar_flops_counter,
    "sigmoid": pool_flops_counter,
    "sigmoid_v2": pool_flops_counter,
    # add
    "bias_add": bias_add_flops_counter,
    "add_n": add_n_flops_counter,
    # matmul
    "matmul": matmul_flops_counter,
    "broadcast_matmul": matmul_flops_counter,
    # norm
    "normalization": normalization_flops_counter,
    # scalar
    "scalar_mul": scalar_flops_counter,
    "scalar_add": scalar_flops_counter,
    "scalar_sub": scalar_flops_counter,
    "scalar_div": scalar_flops_counter,
    # stats
    "var": empty_flops_counter,
    # math
    "sqrt": empty_flops_counter,
    "reduce_sum": reduce_flops_counter,
    # broadcast
    "broadcast_mul": broadcast_flops_counter,
    "broadcast_add": broadcast_flops_counter,
    "broadcast_sub": broadcast_flops_counter,
    "broadcast_div": broadcast_flops_counter,
    # empty
    "reshape": empty_flops_counter,
    "ones_like": empty_flops_counter,
    "zero_like": empty_flops_counter,
    "flatten": empty_flops_counter,
    "concat": empty_flops_counter,
    "transpose": empty_flops_counter,
    "slice": empty_flops_counter,
}

MODULES_MAPPING = {
    # convolutions
    nn.Conv1d: conv_flops_counter_hook,
    nn.Conv2d: conv_flops_counter_hook,
    nn.Conv3d: conv_flops_counter_hook,
    # activations
    nn.ReLU: relu_flops_counter_hook,
    nn.PReLU: relu_flops_counter_hook,
    nn.ELU: relu_flops_counter_hook,
    nn.LeakyReLU: relu_flops_counter_hook,
    nn.ReLU6: relu_flops_counter_hook,
    # poolings
    nn.MaxPool1d: pool_flops_counter_hook,
    nn.AvgPool1d: pool_flops_counter_hook,
    nn.AvgPool2d: pool_flops_counter_hook,
    nn.MaxPool2d: pool_flops_counter_hook,
    nn.MaxPool3d: pool_flops_counter_hook,
    nn.AvgPool3d: pool_flops_counter_hook,
    # nn.AdaptiveMaxPool1d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool1d: pool_flops_counter_hook,
    # nn.AdaptiveMaxPool2d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool2d: pool_flops_counter_hook,
    # nn.AdaptiveMaxPool3d: pool_flops_counter_hook,
    nn.AdaptiveAvgPool3d: pool_flops_counter_hook,
    # BNs
    nn.BatchNorm1d: bn_flops_counter_hook,
    nn.BatchNorm2d: bn_flops_counter_hook,
    nn.BatchNorm3d: bn_flops_counter_hook,
    # INs
    nn.InstanceNorm1d: in_flops_counter_hook,
    nn.InstanceNorm2d: in_flops_counter_hook,
    nn.InstanceNorm3d: in_flops_counter_hook,
    # GN TODO(hujiakui)
    nn.GroupNorm: bn_flops_counter_hook,
    # FC
    nn.Linear: linear_flops_counter_hook,
    # Upscale
    nn.Upsample: upsample_flops_counter_hook,
    # Deconvolution
    nn.ConvTranspose1d: conv_flops_counter_hook,
    nn.ConvTranspose2d: conv_flops_counter_hook,
    nn.ConvTranspose3d: conv_flops_counter_hook,
    # RNN
    nn.RNN: rnn_flops_counter_hook,
    nn.GRU: rnn_flops_counter_hook,
    nn.LSTM: rnn_flops_counter_hook,
    nn.RNNCell: rnn_cell_flops_counter_hook,
    nn.LSTMCell: rnn_cell_flops_counter_hook,
    nn.GRUCell: rnn_cell_flops_counter_hook,
    # nn.MultiheadAttention: multihead_attention_counter_hook
}

if hasattr(nn, 'GELU'):
    MODULES_MAPPING[nn.GELU] = relu_flops_counter_hook
