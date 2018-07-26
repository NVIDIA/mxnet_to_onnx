#  Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
# 
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the 
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its 
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
# 
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY 
#  EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
#  IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
#  PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
#  CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
#  EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, 
#  PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
#  PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
#  OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE 
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
mx_to_uff_converter_functions.py

Conversion Functions for common layers.
Add new functions here with a decorator.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import defs, checker, helper, numpy_helper, mapping

from .mx2onnx_converter import MxNetToONNXConverter as mx2onnx

import numpy as np

import re

import sys 

def looks_like_weight(name):
    """Internal helper to figure out if node should be hidden with `hide_weights`.
    """
    if name.endswith("_weight"):
        return True
    if name.endswith("_bias"):
        return True
    if name.endswith("_beta") or name.endswith("_gamma") or name.endswith("_moving_var") or name.endswith("_moving_mean"):
        return True
    return False


@mx2onnx.register("null")
def convert_weights_and_inputs(node, **kwargs):
    name = node["name"]
    if looks_like_weight(name):
        weights = kwargs["weights"]
        initializer = kwargs["initializer"]
        weights = kwargs["weights"]
        np_arr = weights[name]
        data_type = mapping.NP_TYPE_TO_TENSOR_TYPE[np_arr.dtype]
        dims = np.shape(np_arr)

        tensor_node = helper.make_tensor_value_info(name, data_type, dims)

        initializer.append(
            helper.make_tensor(
                name=name, 
                data_type=data_type,
                dims=dims,
                vals=np_arr.flatten().tolist(),
                raw=False,
            )
        )

        return tensor_node
    else:
        tval_node = helper.make_tensor_value_info(name, kwargs["in_type"], kwargs["in_shape"])
        return tval_node


@mx2onnx.register("Deconvolution")
def convert_deconvolution(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]

    num_inputs = len(inputs)

    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[inputs[0][0]].name
    weights_node = proc_nodes[inputs[1][0]].name

    if num_inputs > 2:
        bias_node = proc_nodes[inputs[2][0]].name

    attrs = node.get("attrs")
    tuple_re = re.compile('\([0-9|,| ]+\)')

    def parse_helper(attrs_name, alt_value=None):
        if attrs is None:
            return alt_value
        attrs_str = attrs.get(attrs_name)
        if attrs_str is None:
            return alt_value
        attrs_match = tuple_re.search(attrs_str)
        if attrs_match is not None:
            if attrs_match.span() == (0, len(attrs_str)):
                dims = eval(attrs_str)
                return dims
            else:
                raise AttributeError("Malformed %s dimensions: %s" % (attrs_name, str(attrs_str)))
        return alt_value

    num_filter = int(attrs["num_filter"])
    kernel_dims = list(parse_helper("kernel"))
    stride_dims = list(parse_helper("stride", [1, 1]))
    pad_dims = parse_padding(attrs)
    num_group = int(attrs.get("num_group", 1))

    # Not sure why this is included, it seems to change what the graphs is doing.
    # TODO(kellens): Ask Marek if this is requried.
    # if len(pad_dims) < 2 * len(kernel_dims):
    #     pad_dims = [0] * (2 * len(kernel_dims) - len(pad_dims)) + pad_dims

    input_nodes = [input_node, weights_node]
    if num_inputs > 2:
        input_nodes.append(bias_node)

    deconv_node = helper.make_node(
        "ConvTranspose",
        inputs=input_nodes,
        outputs=[name],
        kernel_shape=kernel_dims,
        strides=stride_dims,
        pads=pad_dims,
        group=num_group,
        name=name
    )

    return deconv_node


    @mx2onnx.register("Convolution")
def convert_convolution(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]

    num_inputs = len(inputs)

    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[inputs[0][0]].name
    weights_node = proc_nodes[inputs[1][0]].name

    if num_inputs > 2:
        bias_node = proc_nodes[inputs[2][0]].name

    attrs = node.get("attrs")
    tuple_re = re.compile('\([0-9|,| ]+\)')

    def parse_helper(attrs_name, alt_value=None):
        if attrs is None:
            return alt_value
        attrs_str = attrs.get(attrs_name)
        if attrs_str is None:
            return alt_value
        attrs_match = tuple_re.search(attrs_str)
        if attrs_match is not None:
            if attrs_match.span() == (0, len(attrs_str)):
                dims = eval(attrs_str)
                return dims
            else:
                raise AttributeError("Malformed %s dimensions: %s" % (attrs_name, str(attrs_str)))
        return alt_value

    num_filter = int(attrs["num_filter"])
    kernel_dims = list(parse_helper("kernel"))
    stride_dims = list(parse_helper("stride", [1, 1]))
    pad_dims = parse_padding(attrs)
    num_group = int(attrs.get("num_group", 1))

    # Not sure why this is included, it seems to change what the graphs is doing.
    # TODO(kellens): Ask Marek if this is requried.
    # if len(pad_dims) < 2 * len(kernel_dims):
    #     pad_dims = [0] * (2 * len(kernel_dims) - len(pad_dims)) + pad_dims

    input_nodes = [input_node, weights_node]
    if num_inputs > 2:
        input_nodes.append(bias_node)

    conv_node = helper.make_node(
        "Conv",
        inputs=input_nodes, 
        outputs=[name],
        kernel_shape=kernel_dims,
        strides=stride_dims, 
        pads=pad_dims, 
        group=num_group,
        name=name,
        )

    return conv_node


@mx2onnx.register("FullyConnected")
def convert_fully_connected(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]
    input_node_id = inputs[0][0]
    weight_node_id = inputs[1][0]
    bias_node_id = inputs[2][0]    
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_node_id]
    weights_node = proc_nodes[weight_node_id]
    bias_node = proc_nodes[bias_node_id]

    input_name = input_node.name
    weights_name = weights_node.name
    bias_name = bias_node.name 

    node = helper.make_node(
        "Gemm",
        [input_name, weights_name, bias_name],  # input (A, B, C) - C can be in place
        [name],  # output
        alpha=1.0,
        beta=1.0,
        transA=False,
        transB=True,
        name=name
    ) 

    return node

@mx2onnx.register("BatchNorm")
def convert_batchnorm(node, **kwargs):
    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]

    attrs = node["attrs"]
    # Default momentum is 0.9
    try:
        momentum = float(attrs["momentum"])
    except:
        momentum = 0.9
        # Default eps is 0.001
    try:
        eps = float(attrs["eps"])
    except:
        eps = 0.001

    data_idx = inputs[0][0]
    gamma_idx = inputs[1][0]
    beta_idx = inputs[2][0]
    moving_mean_idx = inputs[3][0]
    moving_var_idx = inputs[4][0]

    data_node = proc_nodes[data_idx].name
    gamma_node = proc_nodes[gamma_idx].name
    beta_node = proc_nodes[beta_idx].name

    mov_mean_node = proc_nodes[moving_mean_idx]
    mov_mean_node = mov_mean_node.name
    mov_var_node = proc_nodes[moving_var_idx].name

    bn_node = helper.make_node(
        "BatchNormalization",
        [data_node,
         gamma_node,  # scale
         beta_node,  # bias
         mov_mean_node,
         mov_var_node
        ],
        [name],
        name=name,
        epsilon=eps,
        momentum=momentum,
        is_test=1,
        spatial=1,
        consumed_inputs=(0, 0, 0, 1, 1)
    )

    return bn_node


@mx2onnx.register("Activation")
def convert_activation(node, **kwargs):
    name = node["name"]

    proc_nodes = kwargs["proc_nodes"]
    attrs = node["attrs"]
    act_type = attrs["act_type"]
    
    inputs = node["inputs"]
    input_node_idx = inputs[0][0]
    input_node = proc_nodes[input_node_idx].output[0]

    # Creating a dictionary here, but if this titlecase pattern
    # is consistent for other activations, this can be changed to
    # mxnet_name.title()
    act_types = {
        "tanh": "Tanh",
        "relu": "Relu",
        "sigmoid": "Sigmoid",
        "softrelu": "Softplus",
        "softsign": "Softsign"
    }

    act_name = act_types.get(act_type)
    if act_name:
        node = helper.make_node(
            act_name,
            [input_node],
            [name],
            name=name
        )
    else: 
        raise AttributeError(
            "Activation %s not implemented or recognized in the converter" % act_type
        )

    return node
 

def parse_padding(attrs):
    tuple_re = re.compile('\([0-9|,| ]+\)')

    def parse_helper(attrs_name, alt_value=None):
        if attrs is None:
            return alt_value
        attrs_str = attrs.get(attrs_name)
        if attrs_str is None:
            return alt_value
        attrs_match = tuple_re.search(attrs_str)
        if attrs_match is not None:
            if attrs_match.span() == (0, len(attrs_str)):
                dims = eval(attrs_str)
                return dims
            else:
                raise AttributeError("Malformed %s dimensions: %s" % (attrs_name, str(attrs_str)))
        return alt_value

    symetric_pads = list(parse_helper("pad", [0, 0]))
    result = []

    # Each padding in MXNet is assumed to be symetric in dim1, dim2 ...
    # In ONNX we need to have a start_dim1, start_dim2, ..., end_dim1, end_dim2
    for pad in symetric_pads:
        result.append(pad)
    for pad in symetric_pads:
        result.append(pad)
    return result


@mx2onnx.register("Pooling")
def convert_pooling(node, **kwargs):
    proc_nodes = kwargs["proc_nodes"]
    attrs = node["attrs"]
    kernel = eval(attrs["kernel"])
    pool_type = attrs["pool_type"]

    # Default stride in MXNet for pooling is (1,1)
    stride = eval(attrs["stride"]) if attrs.get("stride") else (1, 1)

    # Global pooling is set explicitly with an attr on the op.
    global_pool = eval(attrs["global"]) if attrs.get("global") else None

    node_inputs = node["inputs"]    
    input_node_idx = node_inputs[0][0]
    input_node = proc_nodes[input_node_idx]
    name = node["name"]

    pad_dims = parse_padding(attrs)

    pool_types = {"max": "MaxPool", "avg": "AveragePool"}
    global_pool_types = {"max": "GlobalMaxPool", "avg": "GlobalAveragePool"}

    if global_pool:
        node = helper.make_node(
            global_pool_types[pool_type],
            [input_node.output[0]],
            [name],
            name=name,
            pads=pad_dims
        )
    else:
        node = helper.make_node(
            pool_types[pool_type],
            [input_node.output[0]],  # input
            [name],
            # dilations = [0, 0],
            kernel_shape=kernel,
            pads=pad_dims,
            strides=stride,
            name=name
        )
    return node


@mx2onnx.register("exp")
def convert_exp(node, **kwargs):
    raise NotImplementedError


# There's also mx.sym.softmax(), which doesn't do cross-entropy loss,
# just softmax for inference - hence the name convert_softmax_output.
@mx2onnx.register("SoftmaxOutput")
def convert_softmax_output(node, **kwargs):
#    print("\nIn convert_softmax_output")
    inputs = node["inputs"]
    input1_idx = inputs[0][0]
    proc_nodes = kwargs["proc_nodes"]
    input1 = proc_nodes[input1_idx]
    name = node["name"]
  
    softmax_node = helper.make_node(
        "Softmax",
        [input1.output[0]],
        [name],
        axis=1,
        name=name
    )

    return softmax_node


@mx2onnx.register("Crop")
def convert_concat(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]
    proc_nodes = kwargs["proc_nodes"]
    input_names = [proc_nodes[i[0]].name for i in inputs]
    attrs = node["attrs"]
    border = [0, 0, 0, 0]
    offset = list(eval(attrs['offset']))
    if len(inputs) == 2:
        border = inputs[1]
    axis = int(node.get("attrs", {}).get("axis", 1))
    concat_node = helper.make_node(
        "Crop",
        input_names,
        [name],
        border=border,
        scale=offset,
        name=name
    )
    return concat_node

@mx2onnx.register("Concat")
def convert_concat(node, **kwargs):
    name = node["name"]
    inputs = node["inputs"]
    proc_nodes = kwargs["proc_nodes"]
    input_names = [proc_nodes[i[0]].name for i in inputs]
    axis = int(node.get("attrs", {}).get("axis", 1))
    concat_node = helper.make_node(
        "Concat",
        input_names,
        [name],
        axis = axis,
        name = name
    )
    return concat_node

@mx2onnx.register("Dropout")
def convert_dropout(node, **kwargs):
    name = node["name"]
    input_id = node["inputs"][0][0]
    input_name = kwargs["proc_nodes"][input_id].name
    attrs = node["attrs"]
    p = float(attrs["p"])
    dropout_node = helper.make_node(
        "Dropout",
        [input_name],
        [name],
        ratio = p,
        is_test = 0,
        name = name
    )
    return dropout_node

@mx2onnx.register("Flatten")
def convert_flatten(node, **kwargs):
    name = node["name"]
    input_idx = node["inputs"][0][0]
    proc_nodes = kwargs["proc_nodes"]
    input_node = proc_nodes[input_idx].name #.output[0]

    flatten_node = helper.make_node(
        "Flatten",
        [input_node],
        [name],
        name = name,
        axis = 1
    )
    return flatten_node

@mx2onnx.register("_mul_scalar")
def convert_mul_scalar(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("elemwise_add")
def convert_elementwise_add(node, **kwargs):

    name = node["name"]
    proc_nodes = kwargs["proc_nodes"]
    inputs = node["inputs"]
    weights = kwargs["weights"]

    a = inputs[0][0]
    b = inputs[1][0]

    a_node = proc_nodes[a].name
    b_node = proc_nodes[b].name

    add_node = helper.make_node(
        "Add",
        [a_node, b_node],
        [name],
        name = name,
    )
  
    return add_node

@mx2onnx.register("_sub")
def convert_elementwise_sub(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("abs")
def convert_abs(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("_mul")
def convert_mul(node, proc_nodes):
    raise NotImplementedError


@mx2onnx.register("_div")
def convert_div(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("log")
def convert_log(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("max")
def convert_max(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("_maximum")
def convert_maximum(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("min")
def convert_min(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("_minimum")
def convert_minimum(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("_power")
def convert_power(node, **kwargs):
    raise NotImplementedError


@mx2onnx.register("sqrt")
def convert_sqrt(node, **kwargs):
    raise NotImplementedError
