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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

import sys

from onnx import (defs, checker, helper, numpy_helper, mapping, onnx_pb2,
                  ModelProto, GraphProto, NodeProto, AttributeProto, TensorProto)

from onnx.helper import make_tensor, make_tensor_value_info

class MxNetToONNXConverter:

    registry_ = {}
    input_output_maps_ = {}

    def __init__(self):
        # topologically sorted nodes
        self.nodes = []
        self.input_tensors = []
        self.output_tensors = []

    @staticmethod
    def register(op_name):

        def wrapper(func):
            MxNetToONNXConverter.registry_[op_name] = func
            return func

        return wrapper

    @staticmethod
    def convert_layer(node, **kwargs):
        op = str(node["op"])
        if op not in MxNetToONNXConverter.registry_:
             raise AttributeError("No conversion function registered for op type %s yet." % op)
        convert_fun = MxNetToONNXConverter.registry_[op]
        return convert_fun(node, **kwargs)

    # Add transpose?
    @staticmethod
    def convert_weights_to_numpy(weights_dict):
        return dict([(k.replace("arg:", "").replace("aux:", ""), v.asnumpy()) for k, v in weights_dict.items()])
  
    def convert_mx2onnx_graph(self, mx_graph, mx_weights, in_shape, in_type, log=False):
        print("\nconverting weights from MxNet NDArrays to NumPy arrays.\n")
        weights = MxNetToONNXConverter.convert_weights_to_numpy(mx_weights)

        onnx_graph = GraphProto() 

        initializer = []
        all_processed_nodes = []
        onnx_processed_nodes = []   
        onnx_processed_inputs = []
        onnx_processed_outputs = []

        for idx, node in enumerate(mx_graph):
           op = node["op"]
           name = node["name"]
           if log:
               print("Converting idx: %d, op: %s, name: %s" % (idx, op, name))
           converted = MxNetToONNXConverter.convert_layer(
               node,
               mx_graph = mx_graph,
               weights = weights,
               in_shape = in_shape,
               in_type = in_type,
               proc_nodes = all_processed_nodes,
               initializer = initializer
           )

           if isinstance(converted, onnx_pb2.ValueInfoProto):
               if idx < (len(mx_graph) - 1):
                   onnx_processed_inputs.append(converted)
               else:
                   onnx_processed_outputs.append(converted)
           elif isinstance(converted, onnx_pb2.NodeProto):
               if idx < (len(mx_graph) - 1):
                   onnx_processed_nodes.append(converted)
               else:
                   onnx_processed_nodes.append(converted)
                   onnx_processed_outputs.append(
                       make_tensor_value_info(
                           name=converted.name,
                           elem_type=mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype('float32')],
                           shape=(in_shape[0], -1)
                       )
                   )
                   if log:
                       print("Output node is: %s" % converted.name)
           elif isinstance(converted, onnx_pb2.TensorProto):
               raise ValueError("Did not expect TensorProto")
               if idx < (len(mx_graph) - 1):
                   onnx_processed_inputs.append(converted)
               else:
                   onnx_processed_outputs.append(converted)
           else:
               print(converted)
               raise ValueError("node is of an unrecognized type: %s" % type(node))             
               
           all_processed_nodes.append(converted)

        graph = helper.make_graph(
            onnx_processed_nodes,
            "main",
            onnx_processed_inputs,
            onnx_processed_outputs
        )

        graph.initializer.extend(initializer)

        checker.check_graph(graph)
        return graph

