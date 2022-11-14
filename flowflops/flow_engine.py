"""
Copyright (C) 2021 Sovrasov V. - All Rights Reserved
 * You may use, distribute and modify this code under the
 * terms of the MIT license.
 * You should have received a copy of the MIT license with
 * this file. If not visit https://opensource.org/licenses/MIT
"""

import re
import sys
from functools import partial

import oneflow as flow
import oneflow.nn as nn

from flowflops.flow_ops import CUSTOM_MODULES_MAPPING, MODULES_MAPPING, GRAPH_FLOPS_COUNT_FUNC
from flowflops.utils import flops_to_string, params_to_string


def get_graph_flops(
    model,
    input_res,
    print_per_layer_stat=True,
    input_constructor=None, 
    ost=sys.stdout,
    verbose=False, 
    ignore_modules=[],
    custom_modules_hooks={},
    output_precision=3,
    flops_units='GMac',
    param_units='M'
):
    _, params = get_eager_flops(
        model,
        input_res,
        input_constructor=input_constructor, 
        ost=ost, 
        ignore_modules=ignore_modules,
        custom_modules_hooks=custom_modules_hooks,
        output_precision=output_precision,
        flops_units=flops_units,
        param_units=param_units,
        verbose=False,
        print_per_layer_stat=False
    )
    model.eval()

    class ModelGraph(nn.Graph):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def build(self, input):
            return self.model(input)

    # build graph
    x = flow.rand(input_res).to(next(model.parameters()).device)
    graph = ModelGraph(model)
    _ = graph(x)

    # init
    graph_str = repr(graph)
    p_size = re.compile(r"size=\(.*?\)", re.S)
    op_in = re.compile(r"\(.*?\) -> ", re.S)
    op_name_conf = re.compile(r"\)\), (.*?):", re.S)
    op_name2op_shape = {}

    # get the shape of in-tensors
    data = re.finditer("OPERATOR:.*", graph_str)
    for i in data:
        op_str = re.findall(op_in, i.group())[0].replace(" -> ", "").replace("[", "")
        if op_str == "()":
            continue
        op_str = ")), " + op_str.replace("(", "", 1)
        op_names = re.findall(op_name_conf, op_str)
        size_strs = re.findall(p_size, op_str)
        for j, size_str in enumerate(size_strs):
            size_attr = size_str.replace("size=", "")
            if size_attr[-2] == ",":
                size_attr = size_attr.replace(",", "")
            data_size = tuple(map(int, size_attr[1:-1].split(", ")))
            op_name2op_shape[op_names[j]] = data_size

    # calculate flops and params
    flops: int = 0
    forward_ops = graph._forward_job_proto.net.op
    for op in forward_ops:
        if op.WhichOneof("op_type") == "user_conf":
            attr = op.user_conf.attr
            input_strs = op.user_conf.input
            op_type_name = op.user_conf.op_type_name
            flops += GRAPH_FLOPS_COUNT_FUNC[op_type_name](attr, input_strs, op_name2op_shape)

    return flops, params


def get_eager_flops(
    model, 
    input_res,
    print_per_layer_stat=True,
    input_constructor=None, 
    ost=sys.stdout,
    verbose=False, 
    ignore_modules=[],
    custom_modules_hooks={},
    output_precision=3,
    flops_units='GMac',
    param_units='M'
):
    global CUSTOM_MODULES_MAPPING
    CUSTOM_MODULES_MAPPING = custom_modules_hooks
    flops_model = add_flops_counting_methods(model)
    flops_model.eval()
    flops_model.start_flops_count(
        ost=ost, 
        verbose=verbose,
        ignore_list=ignore_modules
    )
    if input_constructor is not None:
        input = input_constructor(input_res)
        flops_model(**input)
    else:
        try:
            batch = flow.ones(()).new_empty(
                input_res,
                dtype=next(flops_model.parameters()).dtype,
            )
        except StopIteration:
            batch = flow.ones(()).new_empty(input_res)

        flops_model(batch.to(next(flops_model.parameters()).device))

    flops_count, params_count = flops_model.compute_average_flops_cost()
    if print_per_layer_stat:
        print_model_with_flops(
            flops_model,
            flops_count,
            params_count,
            ost=ost,
            flops_units=flops_units,
            param_units=param_units,
            precision=output_precision
        )
    flops_model.stop_flops_count()
    CUSTOM_MODULES_MAPPING = {}

    return flops_count, params_count


def accumulate_flops(self):
    if is_supported_instance(self):
        return self.__flops__
    else:
        sum = 0
        for m in self.children():
            sum += m.accumulate_flops()
        return sum


def print_model_with_flops(
    model, 
    total_flops, 
    total_params, 
    flops_units='GMac',
    param_units='M', 
    precision=3, 
    ost=sys.stdout
):
    if total_flops < 1:
        total_flops = 1
    if total_params < 1:
        total_params = 1

    def accumulate_params(self):
        if is_supported_instance(self):
            return self.__params__
        else:
            sum = 0
            for m in self.children():
                sum += m.accumulate_params()
            return sum

    def flops_repr(self):
        accumulated_params_num = self.accumulate_params()
        accumulated_flops_cost = self.accumulate_flops() / model.__batch_counter__
        return ', '.join([
            params_to_string(
                accumulated_params_num,
                units=param_units, precision=precision
            ),
            '{:.3%} Params'.format(accumulated_params_num / total_params),
            flops_to_string(
                accumulated_flops_cost,
                units=flops_units, precision=precision
            ),
            '{:.3%} MACs'.format(accumulated_flops_cost / total_flops),
            self.original_extra_repr()
        ])

    def add_extra_repr(m):
        m.accumulate_flops = accumulate_flops.__get__(m)
        m.accumulate_params = accumulate_params.__get__(m)
        flops_extra_repr = flops_repr.__get__(m)
        if m.extra_repr != flops_extra_repr:
            m.original_extra_repr = m.extra_repr
            m.extra_repr = flops_extra_repr
            assert m.extra_repr != m.original_extra_repr

    def del_extra_repr(m):
        if hasattr(m, 'original_extra_repr'):
            m.extra_repr = m.original_extra_repr
            del m.original_extra_repr
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    model.apply(add_extra_repr)
    # print(repr(model), file=ost)
    model.apply(del_extra_repr)


def get_model_parameters_number(model):
    params_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params_num


def add_flops_counting_methods(net_main_module):
    # adding additional methods to the existing module object,
    # this is done this way so that each function has access to self object
    net_main_module.start_flops_count = start_flops_count.__get__(net_main_module)
    net_main_module.stop_flops_count = stop_flops_count.__get__(net_main_module)
    net_main_module.reset_flops_count = reset_flops_count.__get__(net_main_module)
    net_main_module.compute_average_flops_cost = compute_average_flops_cost.__get__(net_main_module)

    net_main_module.reset_flops_count()

    return net_main_module


def compute_average_flops_cost(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Returns current mean flops consumption per image.
    """

    for m in self.modules():
        m.accumulate_flops = accumulate_flops.__get__(m)

    flops_sum = self.accumulate_flops()

    for m in self.modules():
        if hasattr(m, 'accumulate_flops'):
            del m.accumulate_flops

    params_sum = get_model_parameters_number(self)
    return flops_sum / self.__batch_counter__, params_sum


def start_flops_count(self, **kwargs):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Activates the computation of mean flops consumption per image.
    Call it before you run the network.
    """
    add_batch_counter_hook_function(self)

    seen_types = set()

    def add_flops_counter_hook_function(module, ost, verbose, ignore_list):
        if type(module) in ignore_list:
            seen_types.add(type(module))
            if is_supported_instance(module):
                module.__params__ = 0
        elif is_supported_instance(module):
            if hasattr(module, '__flops_handle__'):
                return
            if type(module) in CUSTOM_MODULES_MAPPING:
                module.register_forward_hook(CUSTOM_MODULES_MAPPING[type(module)])
            elif type(module) in MODULES_MAPPING:
                module.register_forward_hook(MODULES_MAPPING[type(module)])
            else:
                raise RuntimeError
            seen_types.add(type(module))
        else:
            if verbose and not type(module) in (nn.Sequential, nn.ModuleList) and \
               not type(module) in seen_types:
                print('Warning: module ' + type(module).__name__ +
                      ' is treated as a zero-op.', file=ost)
            seen_types.add(type(module))

    self.apply(partial(add_flops_counter_hook_function, **kwargs))


def stop_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Stops computing the mean flops consumption per image.
    Call whenever you want to pause the computation.
    """
    remove_batch_counter_hook_function(self)
    self.apply(remove_flops_counter_hook_function)
    self.apply(remove_flops_counter_variables)


def reset_flops_count(self):
    """
    A method that will be available after add_flops_counting_methods() is called
    on a desired net object.
    Resets statistics computed so far.
    """
    add_batch_counter_variables_or_reset(self)
    self.apply(add_flops_counter_variable_or_reset)


# ---- Internal functions
def batch_counter_hook(module, input, output):
    batch_size = 1
    if len(input) > 0:
        # Can have multiple inputs, getting the first one
        input = input[0]
        batch_size = len(input)
    else:
        print('Warning! No positional inputs found for a module,'
              ' assuming batch size is 1.')
    module.__batch_counter__ += batch_size


def add_batch_counter_variables_or_reset(module):
    module.__batch_counter__ = 0


def add_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        return

    module.register_forward_hook(batch_counter_hook)


def remove_batch_counter_hook_function(module):
    if hasattr(module, '__batch_counter_handle__'):
        del module.__batch_counter_handle__


def add_flops_counter_variable_or_reset(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__') or hasattr(module, '__params__'):
            print('Warning: variables __flops__ or __params__ are already '
                  'defined for the module' + type(module).__name__ +
                  ' flowflops can affect your code!')
            module.__flowflops_backup_flops__ = module.__flops__
            module.__flowflops_backup_params__ = module.__params__
        module.__flops__ = 0
        module.__params__ = get_model_parameters_number(module)


def is_supported_instance(module):
    if type(module) in MODULES_MAPPING or type(module) in CUSTOM_MODULES_MAPPING:
        return True
    return False


def remove_flops_counter_hook_function(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops_handle__'):
            del module.__flops_handle__


def remove_flops_counter_variables(module):
    if is_supported_instance(module):
        if hasattr(module, '__flops__'):
            del module.__flops__
            if hasattr(module, '__flowflops_backup_flops__'):
                module.__flops__ = module.__flowflops_backup_flops__
        if hasattr(module, '__params__'):
            del module.__params__
            if hasattr(module, '__flowflops_backup_params__'):
                module.__params__ = module.__flowflops_backup_params__
