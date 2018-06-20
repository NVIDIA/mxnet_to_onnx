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

import os.path
import subprocess
from unittest import TestCase

import mxnet as mx
import numpy as np

# needed by both the exporter and importer
import onnx

# MxNet exporter
from mx2onnx_converter.conversion_helpers import from_mxnet

# MxNet importer
# Needed for ONNX -> NNVM -> MxNet conversion
# to validate the results of the export
#import onnx_mxnet
from mxnet.contrib.onnx import import_model

def check_gpu_id(gpu_id):
    try:
        result = subprocess.check_output("nvidia-smi --query-gpu=gpu_bus_id --format=csv,noheader", shell=True).strip()
    except OSError as e:
        return False
    if not isinstance(result, str):
        result = str(result.decode("ascii"))
    gpu_ct = len(result.split("\n"))
    # count is zero-based
    exists = gpu_id < gpu_ct
    print("\nChecked for GPU ID %d. Less than GPU count (%d)? %s\n" % (gpu_id, gpu_ct, exists)) 
    return exists

# MxNet LeNet-5 implementation
def lenet5():
    data = mx.sym.var('data')
    # first conv layer
    conv1 = mx.sym.Convolution(data=data, kernel=(5,5), num_filter=20)
    tanh1 = mx.sym.Activation(data=conv1, act_type="tanh")
    pool1 = mx.sym.Pooling(data=tanh1, pool_type="max", kernel=(2,2), stride=(2,2))
    # second conv layer
    conv2 = mx.sym.Convolution(data=pool1, kernel=(5,5), num_filter=50)
    tanh2 = mx.sym.Activation(data=conv2, act_type="tanh")
    pool2 = mx.sym.Pooling(data=tanh2, pool_type="max", kernel=(2,2), stride=(2,2))
    # first fullc layer
    flatten = mx.sym.flatten(data=pool2)
    fc1 = mx.symbol.FullyConnected(data=flatten, num_hidden=500)
    tanh3 = mx.sym.Activation(data=fc1, act_type="tanh")
    # second fullc
    fc2 = mx.sym.FullyConnected(data=tanh3, num_hidden=10)
    # softmax loss
    lenet = mx.sym.SoftmaxOutput(data=fc2, name='softmax')
    return lenet

# train LeNet-5 model on MNIST data
def train_lenet5(num_epochs, gpu_id, train_iter, val_iter, test_iter, batch_size):
    ctx = mx.gpu(gpu_id) if gpu_id is not None else mx.cpu()
    print("\nUsing %s to train" % str(ctx))
    lenet_model = lenet5()
    lenet_model = mx.mod.Module(lenet_model, context=ctx)
    # This is cached so download will only take place if needed
    mnist = mx.test_utils.get_mnist()
    train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
    val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    data = mx.sym.var('data')
    data = mx.sym.flatten(data=data)

    lenet_model.fit(train_iter,
                    eval_data=val_iter,
                    optimizer='sgd',
                    optimizer_params={'learning_rate': 0.1, 'momentum': 0.9},
                    eval_metric='acc',
                    batch_end_callback = mx.callback.Speedometer(batch_size, 100),
                    num_epoch=num_epochs)

    test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)

    # predict accuracy for lenet
    acc = mx.metric.Accuracy()
    lenet_model.score(test_iter, acc)
    accuracy = acc.get()[1]
    print("Training accuracy: %.2f" % accuracy)
    assert accuracy > 0.98, "Accuracy was too low"
    return lenet_model

class LeNet5Test(TestCase):

    def __init__(self, *args, **kwargs):
        TestCase.__init__(self, *args, **kwargs)
#        self.tearDown = lambda: subprocess.call("rm -f *.gz *-symbol.json *.params *.onnx", shell=True)

    def test_convert_and_compare_prediction(self):
        # get data iterators and set basic hyperparams
        num_epochs = 10
        mnist = mx.test_utils.get_mnist()
        batch_size = 1000
        train_iter = mx.io.NDArrayIter(mnist['train_data'], mnist['train_label'], batch_size, shuffle=True)
        val_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
        test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
        model_name = 'lenet5'
        model_file = '%s-symbol.json' % model_name
        params_file = '%s-%04d.params' % (model_name, num_epochs)
        onnx_file = "%s.onnx" % model_name
        test_gpu_id = 0
        gpu_id = check_gpu_id(test_gpu_id) 
        if not gpu_id:
            print("\nWARNING: GPU id %d is invalid on this machine" % test_gpu_id)
            gpu_id = None
    
        # If trained model exists, re-use cached version. Otherwise, train model.
        if not (os.path.exists(model_file) and os.path.exists(params_file)):
            print("\n\nTraining LeNet-5 on MNIST data")
            trained_lenet = train_lenet5(num_epochs, gpu_id, train_iter, val_iter, test_iter, batch_size)
            print("Training finished. Saving model")
            trained_lenet.save_checkpoint(model_name, num_epochs)
            # delete object so we can verify correct loading of the checkpoint from disk
            del trained_lenet
        else:
            print("\n\nTrained model exists. Skipping training.")
    
        # Load serialized MxNet model (model-symbol.json + model-epoch.params)
    
        trained_lenet = mx.mod.Module.load(model_name, num_epochs)
        trained_lenet.bind(data_shapes=test_iter.provide_data, label_shapes=None, for_training=False, force_rebind=True)
    
        # Run inference in MxNet from json/params serialized model
        test_iter = mx.io.NDArrayIter(mnist['test_data'], mnist['test_label'], batch_size)
        pred_softmax = trained_lenet.predict(test_iter).asnumpy()
        pred_classes = np.argmax(pred_softmax, axis=1)
    
        # Create and save ONNX model
        print("\nConverting trained MxNet model to ONNX")
        model = from_mxnet(model_file, params_file, [1, 1, 28, 28], np.float32, log=True)
        with open(onnx_file, "wb") as f:
            serialized = model.SerializeToString()
            f.write(serialized)
            print("\nONNX file %s serialized to disk" % onnx_file)
    
        print("\nLoading ONNX file and comparing results to original MxNet output.")
    
        # ONNX load and inference step
        onnx_sym, onnx_arg_params, onnx_aux_params = import_model(onnx_file)
        onnx_mod = mx.mod.Module(symbol=onnx_sym, data_names=['data'], context=mx.cpu(), label_names=None)
    
        # Need to rename data argument from 'data' to 'input_0' because that's how
        # the MxNet ONNX importer expects it by default
        test_iter = mx.io.NDArrayIter(data={'data': mnist['test_data']}, label=None, batch_size=batch_size)
    
        onnx_mod.bind(data_shapes=test_iter.provide_data, label_shapes=None, for_training=False, force_rebind=True)
        onnx_mod.set_params(arg_params=onnx_arg_params, aux_params=onnx_aux_params, allow_missing=True)
    
        onnx_pred_softmax = onnx_mod.predict(test_iter).asnumpy()
        onnx_pred_classes = np.argmax(pred_softmax, axis=1)
    
        pred_matches = onnx_pred_classes == pred_classes
        pred_match_ct = pred_matches.sum()
        pred_total_ct = np.size(pred_matches)
        pct_match = 100.0 * pred_match_ct / pred_total_ct
    
        print("\nOriginal MxNet predictions and ONNX-based predictions after export and re-import:")
        print("Total examples tested: %d" % pred_total_ct)
        print("Matches: %d" % pred_match_ct)
        print("Percent match: %.2f\n" % pct_match)
  
        assert pred_match_ct == pred_total_ct, "Not all predictions from the ONNX representation match"
