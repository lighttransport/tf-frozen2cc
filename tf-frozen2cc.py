# MIT License
# 
# Copyright (c) 2018 Light Transport Entertainment, Inc.
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
# 
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os, sys
import json
import re
import base64

from string import Template

## -------------------------------------------------------------

def topolocal_sort(input_graph):

    from collections import deque

    GRAY, BLACK = 0, 1

    def topological(graph):
        order, enter, state = deque(), set(graph), {}

        def dfs(node):
            state[node] = GRAY
            for k in graph.get(node, ()):
                sk = state.get(k, None)
                if sk == GRAY: raise ValueError("cycle")
                if sk == BLACK: continue
                enter.discard(k)
                dfs(k)
            order.appendleft(node)
            state[node] = BLACK

        while enter: dfs(enter.pop())
        return order

    return topological(input_graph)

    # The MIT License (MIT)
    # Copyright (c) 2014 Alexey Kachayev
    #
    # Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation
    # files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy,
    # modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
    # is furnished to do so, subject to the following conditions:
    #
    # The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
    #
    # THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
    # WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
    # COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
    # ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## -------------------------------------------------------------

def escape_name(name  # type: str
    ):

    # replace '/' deliminer with '___'
    return name.replace('/', '___')


def get_dim_and_len(node):
    shape = node["attr"]["shape"]["shape"]
    n_dim = len(shape["dim"])

    length = 1
    dim = []
    for i in range(n_dim):
        size = int(shape["dim"][i]["size"])
        if size > 1:
            length = length * size
        dim.append(size)

    return (dim, length)

def get_type(node):
    return node["attr"]["T"]["type"]

def get_dtype(node):
    return node["attr"]["dtype"]["type"]

def get_strides(attr):
    strides = []
    assert len(attr["strides"]["list"]["i"]) == 4
    strides.append(int(attr["strides"]["list"]["i"][0]))
    strides.append(int(attr["strides"]["list"]["i"][1]))
    strides.append(int(attr["strides"]["list"]["i"][2]))
    strides.append(int(attr["strides"]["list"]["i"][3]))

    return strides

def get_dilations(attr):
    dilations = []
    assert len(node["dilations"]["list"]["i"]) == 4
    dilations.append(int(node["dilations"]["list"]["i"][0]))
    dilations.append(int(node["dilations"]["list"]["i"][1]))
    dilations.append(int(node["dilations"]["list"]["i"][2]))
    dilations.append(int(node["dilations"]["list"]["i"][3]))

    return dilations

def get_data_format(attr):
    if "s" in attr["data_format"]:
        encoded_str = attr["data_format"]["s"]
        # byte to string(py3)
        return (base64.b64decode(encoded_str)).decode('utf-8')

    raise attr

def get_padding(attr):
    if "s" in attr["padding"]:
        encoded_str = attr["padding"]["s"]
        # byte to string(py3)
        return (base64.b64decode(encoded_str)).decode('utf-8')

    raise attr

def get_dilations(attr):
    if "s" in attr["dilations"]:
        encoded_str = attr["data_format"]["s"]
        # byte to string(py3)
        return (base64.b64decode(encoded_str)).decode('utf-8')

    raise attr

def Abs(op, ctx):
    assert get_type(op) == "DT_FLOAT"

    s = '''
// name: ${name}
static void Abs(std::vector<float> &input, std::vector<float> *output) {

    output->resize(input.size());
    for (size_t i = 0; i < input.size(); i++) {
        (*output)[i] = std::abs(input_a[i]); 
    }
}
'''

    s = Template(s)
    d = op
    return s.substitute(d)

def Add(op, ctx):
    assert get_type(op) == "DT_FLOAT"

    s = '''
// name: ${name}
static void Add(std::vector<float> &input_a, std::vector<float> &input_b, std::vector<float> *output) {

    output->resize(input_a.size());
    for (size_t i = 0; i < input_a.size(); i++) {
        (*output)[i] = input_a[i] + input_b[i]; 
    }
}
'''

    s = Template(s)
    d = op
    return s.substitute(d)

def BiasAdd(op, ctx):
    """
    TODO: Consider `data_format`?
    """

    assert get_type(op) == "DT_FLOAT"

    s = '''
// name: ${name}
static void BiasAdd(std::vector<float> &input_a, std::vector<float> &bias, std::vector<float> *output) {

    output->resize(input_a.size());
    for (size_t i = 0; i < input_a.size(); i++) {
        (*output)[i] = input_a[i] + bias[i]; 
    }
}
'''

    s = Template(s)
    d = op
    return s.substitute(d)


def MatMul(op, ctx):

    #(dim, length) = get_dim_and_len(op)
    assert get_type(op) == "DT_FLOAT"

    # TODO(LTE): Support various dtype.

    s = '''
static void MatMul(std::vector<float> &input_a, std::vector<float> &input_b, std::vector<float> *output) {

    // Assume dimension is 2.
    // TODO(LTE): Support padding.
    // TODO(LTE): Support row major order.
    size_t dim0 = input_a.size();
    size_t dim1 = input_b.size() / dim0;

    output->resize(dim1);
    for (size_t j = 0; j < dim1; j++) {
        double sum = 0.0;
        // dot op
        for (size_t i = 0; i < dim0; i++) {
            // Assume minor_to_major is [0, 1]. i.e. y first then x.
            sum += double(input_a[i] * input_b[i * dim1 + j]);
        }
        (*output)[j] = float(sum);
    }
}
'''
    s = Template(s)

    d = op
    return s.substitute(d)

def Softmax(node, ctx):

    assert get_type(node) == "DT_FLOAT"

    # TODO(LTE): Use log for better numerical accuracy.

    s = '''
static void Softmax(std::vector<float> &input,  std::vector<float> *output) {

    output->resize(input.size());

    double sum = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        sum += double(std::exp(input[i]));
    }
    
    double inv_sum = 1.0 / sum;

    for (size_t i = 0; i < input.size(); i++) {
        (*output)[i] = float(double(std::exp(input[i])) * inv_sum);
    }
}
'''
    s = Template(s)

    d = node
    return s.substitute(d)

def Sigmoid(node, ctx):

    assert get_type(node) == "DT_FLOAT"

    # TODO(LTE): Use log for better numerical accuracy.

    s = '''
static void Sigmoid(std::vector<float> &input,  std::vector<float> *output) {

    output->resize(input.size());

    double sum = 0.0;
    for (size_t i = 0; i < input.size(); i++) {
        (*output)[i] = 1.0 / (1.0 + std::exp(input[i]));
    }
}
'''
    s = Template(s)

    d = node
    return s.substitute(d)

def Reshape(op, ctx):
    raise "TODO"
    pass

def Const(node, ctx):

    ty = node["attr"]["value"]["tensor"]["dtype"] 

    assert (ty == "DT_FLOAT" or ty == "DT_INT32"), "Unsupported type:" + ty

    if not "tensorContent" in node["attr"]["value"]["tensor"]:
        # not a weight node
        return None

    data = node["attr"]["value"]["tensor"]["tensorContent"]
    
    s = '''
static void ConstPrepare_${name_escaped}(std::vector<float> *output) {
    std::string tensorContent = "${data}";
    
    std::string decoded = base64_decode(tensorContent);

    size_t dst_len = decoded.size() / sizeof(float);
    
    output->resize(dst_len);

    const float *src_ptr = reinterpret_cast<const float *>(decoded.data());

    for (size_t i = 0; i < output->size(); i++) {
        (*output)[i] = src_ptr[i];
    } 
}

'''
    st = Template(s)

    d = node
    d["data"] = data
    d["name_escaped"] = escape_name(node["name"])
    return st.substitute(d)

def Placeholder(op, ctx):

    (dim, length) = get_dim_and_len(op)
    assert get_dtype(op) == "DT_FLOAT"

    # TODO(LTE): Support various dtype and dim.

    s = '''
// ${name}
static void PlaceholderPrepare_${name_escaped}(std::vector<float> *output) {
    output->resize(${length});
}

'''
    s = Template(s)

    d = op
    d["length"] = length
    d["name_escaped"] = escape_name(op["name"])
    return s.substitute(d)

def Relu(node, ctx):

    assert get_type(node) == "DT_FLOAT"

    # TODO(LTE): Use log for better numerical accuracy.

    s = '''
static void Relu(std::vector<float> &input,  std::vector<float> *output) {

    output->resize(input.size());

    for (size_t i = 0; i < input.size(); i++) {
        (*output)[i] = std::max(input[i]), 0.0f);
    }
}
'''
    s = Template(s)

    d = node
    return s.substitute(d)

def LRN(op, ctx):
    """
    input: 4D tensor(must be float, bfloat16 or float32 type)
    tf.nn.local response_normalization
    sqr_sum[a, b, c, d] =
    sum(input[a, b, c, d - depth_radius : d + depth_radius + 1] ** 2)
output = input / (bias + alpha * sqr_sum) ** beta

    default:
        depth_radius = 5
        bias = 5
        alpha = 1
        beta = 0.5
    """

    s = '''
static void LRN(std::vector<float> &input,  const int input_shape[4], const int depth_radius, const float bias, const float alpha, const float beta, std::vector<float> *output) {

    output->resize(input.size());

    for (size_t n = 0; n < input_shape[0]; n++) {
        for (size_t h = 0; h < input_shape[2]; h++) {
            for (size_t w = 0; w < input_shape[1]; w++) {
                for (size_t c = 0; c < input_shape[3]; c++) {

                    double sqr_sum = 0.0;

                    for (int k = d - depth_radius; k < d + depth_radius + 1; k++) {
                        size_t src_idx = size_t(std::min(std::max(0, k), input_shape[3]));
                        double s = input[src_idx];
                        s = s * s; // sqr
                        sqr_sum += s;
                    }

                    size_t idx = c * input_shape[0] * input_shape[1] * input_shape[2] + w * input_shape[0] * input_shape[1] + h * input_shape[0];
                    (*output)[idx] = input[idx] / std::pow((bias + alpha * sqr_num), beta);
                }
            }
        }
    }
}
'''
    s = Template(s)

    d = node
    return s.substitute(d)

def StridedSlice(op, ctx):
    raise "TODO"
    pass

def Conv2D_NHWC(op, ctx):
    """
    output[b, i, j, k] =
        sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] * filter[di, dj, q, k]
    Assume strides = [1, stride, stride, 1]
    """
    print(op)
    strides = get_strides(op["attr"])
    assert len(strides) == 4
    assert strides[0] == 1
    assert strides[3] == 1

    # Naiive convolution kernel.
    s = '''
static void Conv2D_NHWC(const std::vector<float> &input, const std::vector<float> &filter, const int input_shape[4], const int filter_shape[4],  const int strides[4], std::vector<float> *output) {
    
    // input: [n, height, width, channels]
    // filter: [filter_height, filter_width, in_channels, out_channels]
    output->resize(input);

    size_t n = input_shape[0];
    size_t width = input_shape[1];
    size_t height = input_shape[2];
    size_t channels = input_shape[3];

    size_t filter_width = filter_shape[1];
    size_t filter_height = filter_shape[2];
    size_t filter_channels = filter_shape[3]; // in_channels

    // Assume n == 1
    // Assume padding == SAME

    for (int i = 0; i < channels; i++) {
        for (int j = 0; j < height; j++) {
            for (int k = 0; k < width; k++) {

                double sum = 0.0;
                for (int q = 0; q < filter_channels; q++) {
                    for (int dj = -filter_height/2; dj <= filter_height/2; dj++) { // height
                        int fy = std::min(std::max(j + dj, 0), height);
                        for (int di = -filter_width/2; di <= filter_width; di++) { // width
                            // clamp
                            int fx = std::min(std::max(k + di, 0), width);

                            size_t src_idx = filter_channels * (fy * width + fx) + q;
                            size_t filter_idx = filter_channels * (fy * filter_width + fx) + q;
                            sum += input[src_idx] * filter[filter_idx];
                        }
                    }
                }

                size_t out_idx = n * (q * width * height+ dj * width) + di;
                (*outout)[out_idx] = sum;
            }
        }
    }
}

'''

    s = Template(s)

    d = op
    #d["length"] = length
    #d["name_escaped"] = escape_name(op["name"])
    return s.substitute(d)

def Conv2D(op, ctx):
    """
    parameters:
        filter
        strides 
        padding
        data_format = 'NHWC'
        dialations = [1,1,1,1]
    """
    data_format = get_data_format(op["attr"])
    assert data_format == 'NHWC', "Unsupported data format " + data_format

    padding = get_padding(op["attr"])
    assert padding == 'SAME', "Unsupported padding " + padding

    print("data_format", data_format)
    print("padding", padding)

    return Conv2D_NHWC(op, ctx)

    pass

def Conv2DBackpropInput(op, ctx):
    raise "TODO"
    pass

def FusedBatchNorm(op, ctx):
    raise "TODO"
    pass

def Pack(op, ctx):
    raise "TODO"
    pass

def Shape(op, ctx):
    raise "TODO"
    pass

def NoOp(op, ctx):
    pass

class Context:
    def __init__(self, nodes):
        self.nodes = nodes
        self.graph = {}

        # Stores information of arguments for layers.
        self.args = {}

        # Assume name is unique
        for node in nodes:
            self.graph[node["name"]] = node 

    def set_arguments(node_name, # type: str
                      args, # type: {}
                     ):
        self.args[node_name] = args


_CodeGenOpTable = {
    'MatMul' : MatMul
  , 'Abs' : Add
  , 'Add' : Add
  , 'BiasAdd' : BiasAdd
  , 'Placeholder' : Placeholder
  , 'Const' : Const
  , 'Softmax' : Softmax
  , 'Sigmoid' : Sigmoid
  , 'StridedSlice' : StridedSlice
  , 'Conv2D' : Conv2D
  , 'Conv2DBackpropInput' : Conv2DBackpropInput
  , 'FusedBatchNorm' : FusedBatchNorm
  , 'Pack' : Pack
  , 'Relu' : Relu
  , 'Shape' : Shape
}

def HasWeightInConst(node):

    ty = node["attr"]["value"]["tensor"]["dtype"] 

    assert (ty == "DT_FLOAT" or ty == "DT_INT32"), "Unsupported type:" + ty

    if "tensorContent" in node["attr"]["value"]["tensor"]:
        return True

    return False

def ConstructArgString(prefix, # type: str
                       node):
    s = ""

    inputs = []

    if "input" in node:
        inputs = node["input"]
    
    for i in range(len(inputs)):
        s += prefix + escape_name(inputs[i])
        if i != (len(inputs) - 1):
            s += ", "

    if len(inputs):
        s += ", "

    # output
    s += "&(" + prefix + escape_name(node["name"]) + ")"

    return s

def EmitBufferDef(deque_graph, # type: collections.deque
            ctx, # type: Context
            ):

    """
    Emit buffer class
    """

    ret = '''
struct Buffer {
'''

    for layer_name in deque_graph:
        node = ctx.graph[layer_name]

        if node["op"] == "NoOp":
            continue

        s = "    std::vector<float> {};\n".format(escape_name(node["name"]))

        ret += s

    ret += '''
};
'''

    return ret


def EmitPrepare(deque_graph, # type: collections.deque
            ctx, # type: Context
            ):

    """
    Emit preparation code(const, reshape)
    """

    ret = ""

    # header
    ret = '''
void NetworkPrepare(Buffer *buffer) {
'''

    for layer_name in deque_graph:
        node = ctx.graph[layer_name]
        print("op", node["op"])

        if node["op"] == "NoOp":
            continue


        if node["op"] == "Const":

            if HasWeightInConst(node):

                # Emit prepare function.
                s = '''
        ConstPrepare_${name_escaped}(${args});
    '''
                st = Template(s)
                d = node
                d["name_escaped"] = escape_name(node["name"])
                d["args"] = ConstructArgString("buffer->", node)

                ret += st.substitute(d)

        elif node["op"] == "Placeholder":

            # Emit prepare function.
            s = '''
    PlaceholderPrepare_${name_escaped}(${args});
'''
            st = Template(s)
            d = node
            d["name_escaped"] = escape_name(node["name"])
            d["args"] = ConstructArgString("buffer->", node)

            ret += st.substitute(d)

    # footer
    ret += '''
}
'''

    return ret

def EmitEval(deque_graph, # type: collections.deque
            ctx, # type: Context
            ):

    """
    Emit evaluation code(eval layers)
    """

    ret = '''
bool NetworkForwardExecute(Buffer *buffer) {
'''

    for layer_name in deque_graph:
        node = ctx.graph[layer_name]

        if node["op"] == "NoOp":
            continue

        elif node["op"] == "Const":
            continue

        elif node["op"] == "Placeholder":
            continue

        s = '''
    ${op}(${args});
'''
        st = Template(s)
        d = node
        d["args"] = ConstructArgString("buffer->", node)

        ret += st.substitute(d)

    ret += '''
    return true;
}
'''

    return ret


def EmitHHeader():
    s = '''
#ifndef LAINNIE_H_
#define LAINNIE_H_

#include <vector>

namespace lainnie {

'''
    return s

def EmitHFooter():
    s = '''

// Prepare buffer for the network
void NetworkPrepare(Buffer *buffer);

// Execute network(forward pass).
bool NetworkForwardExecute(Buffer *buffer);

} // namespace lainnie

#endif // LAINNIE_H_
'''
    return s


def EmitCCHeader(h_filename):
    s = '''
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
'''

    s += "#include \"{}\"\n".format(h_filename)

    s += '''
namespace lainnie {

#include "base64-encdec.inc"

'''
    return s

def EmitCCFooter():
    s = '''
} // namespace lainnie

'''
    return s

def proc(input_filename, # type: str
         output_filename, # type: str
        ):
    j = json.loads(open(input_filename, 'r').read())

    graph = {}
    s = ""

    # Emit layers(ops)
    for (key, c) in j.items():
        if key == "node":

            assert isinstance(c, list)
            ctx = Context(c)

            for node in c:
                name = node['name']
                op_name = node['op']

                if op_name in _CodeGenOpTable:
                    ss = _CodeGenOpTable[op_name](node, ctx)
                    if ss != None:
                        s = s + ss

                graph[name] = []
                    
            # add graph edge
            for node in c:
                if "input" in node:
                    name = node["name"]

                    for i in node["input"]:
                        graph[i].append(name)
    
    # print(graph)

    # topological sort of graph.
    deq = topolocal_sort(graph)

    str_h = EmitBufferDef(deq, ctx)

    # Emit functions
    str_cc = s

    # Emit preparation function
    str_cc += EmitPrepare(deq, ctx)

    # Emit evaluation function
    str_cc += EmitEval(deq, ctx)

    output_filename_h = output_filename + ".h"
    output_filename_cc = output_filename + ".cc"

    with open(output_filename_h, 'w') as f:
        f.write(EmitHHeader())
        f.write(str_h)
        f.write(EmitHFooter())

    with open(output_filename_cc, 'w') as f:
        f.write(EmitCCHeader(os.path.basename(output_filename) + ".h"))
        f.write(str_cc)
        f.write(EmitCCFooter())

def main():
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--input", type=str, required=True, help="Input model in JSON")
  parser.add_argument("--output", type=str, required=True, help="Output C++ base filename")
  args = parser.parse_args()
     
  proc(args.input, args.output)

  print("Done!")

main()
