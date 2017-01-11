MxNet-to-ONNX exporter
==========================


What is this?
------------------


This is the repository for the [MxNet](https://github.com/apache/incubator-mxnet)-to-[ONNX](https://github.com/onnx/onnx) converter, which takes a trained MxNet model, represented in serialized form as the .json/.params file pair, and converts that model to ONNX. Please note that this is a file-to-file conversion - the input is a checkpointed MxNet model, NOT the [NNVM](https://github.com/dmlc/nnvm) graph.

Installation
----------------------------------------

Note that --force will force an upgrade if a previous version was installed. This is equivalent to first uninstalling and then installing again. Without force, an upgrade will not be performed.

```python setup.py install --force```

Also note that since this project depends on ONNX, and ONNX depends on the [Protobuf](https://github.com/google/protobuf) compiler, the installation of the ONNX [pip](https://packaging.python.org/tutorials/installing-packages/#use-pip-for-installing) package will require the compiler. The installation of the native component will depend on your operating system, but on Ubuntu 16.04, you can simply do

```sudo apt-get install protobuf-compiler libprotoc-dev```

See the [details](https://github.com/onnx/onnx/blob/master/README.md) as to what is required to install ONNX. Note that even though the ONNX pip package can be fetched from [PyPI](https://pypi.python.org/pypi), it will still depend on the Protobuf compiler. Hence, even though ONNX is listed in requirements.txt, its installation will depend on the aforementioned native components.


Tests
----------------------------------------

To run the test that:

1. trains [LeNet-5](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf) on [MNIST](http://yann.lecun.com/exdb/mnist/)
2. checkpoints the MxNet a trained model to the .json/.params file pair that represents a serialized MxNet model
3. loads the serialized MxNet model and runs inference on test data
4. converts the serialized MxNet model to ONNX
5. loads the ONNX model and runs inference on test data
6. asserts that all 10,000 predictions match

please run:

```python setup.py test```
 
