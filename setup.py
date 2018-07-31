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

import itertools as it
import os
import re
from setuptools import setup
from subprocess import call
import sys

match_mxnet_req = re.compile(r"mxnet>?=?=\d+.\d+\d*")
extract_major_minor = re.compile(r"\D*(\d+.\d+)\D*")

def check_mxnet_version(min_ver):
    if not int(os.environ.get('UPDATE_MXNET_FOR_ONNX_EXPORTER', '1')):
        print("Env var set to not upgrade MxNet for ONNX exporter. Skipping.")
        return False
    try:
        print("Checking if MxNet is installed.")
        import mxnet as mx
    except ImportError:
        print("MxNet is not installed. Installing version from requirements.txt")
        return False
    ver = float(re.match(extract_major_minor, mx.__version__).group(1))
    min_ver = float(re.match(extract_major_minor, min_ver).group(1))
    if ver < min_ver:
        print("MxNet is installed, but installed version (%s) is older than expected (%s). Upgrading." % (str(ver).rstrip('0'), str(min_ver).rstrip('0')))
        return False
    print("Installed MxNet version (%s) meets the requirement of >= (%s). No need to install." % (str(ver).rstrip('0'), str(min_ver).rstrip('0')))
    return True 

if __name__ == '__main__':

    with open('requirements.txt') as f:
        required = f.read().splitlines()

    mx_match_str = lambda x: re.match(match_mxnet_req, x) is None
    mx_str, new_reqs = tuple([list(i[1]) for i in it.groupby(sorted(required, key = mx_match_str), key = mx_match_str)])
 
    if not check_mxnet_version(mx_str[0]):
        new_reqs += mx_str

    setup(
        install_requires = new_reqs,
        name = 'mx2onnx',
        description = 'MxNet to ONNX converter',
        author = 'NVIDIA Corporation',
        packages = ['mx2onnx_converter'],
        classifiers = [
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3.5' 
        ],
        keywords = 'mxnet onnx',
        zip_safe = False,
        test_suite='nose.collector',
        tests_require=['nose'],
        version = '0.1'
    )
   
    call("rm -rf dist".split())
    call("rm -rf build".split()) 
    call("rm -rf mx2onnx.egg-info".split())
