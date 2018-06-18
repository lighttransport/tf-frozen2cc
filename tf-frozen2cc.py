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

def Const(node, ctx):

    ty = node["attr"]["value"]["tensor"]["dtype"] 

    assert ty == "DT_FLOAT"

    data = node["attr"]["value"]["tensor"]["tensorContent"]
    
    s = '''
static void ConstInit_${name}(std::vector<float> *output) {
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
    return st.substitute(d)

def Placeholder(op, ctx):

    (dim, length) = get_dim_and_len(op)
    assert get_dtype(op) == "DT_FLOAT"

    # TODO(LTE): Support various dtype and dim.

    s = '''
// ${name}
static void PlaceholderInit_${name}(std::vector<float> *output) {
    output->resize(${length});
}

'''
    s = Template(s)

    d = op
    d["length"] = length
    return s.substitute(d)

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
  , 'Add' : Add
  , 'Placeholder' : Placeholder
  , 'Const' : Const
  , 'Softmax' : Softmax
}

def ConstructArgString(prefix, # type: str
                       node):
    s = ""

    inputs = []

    if "input" in node:
        inputs = node["input"]
    
    for i in range(len(inputs)):
        s += prefix + inputs[i]
        if i != (len(inputs) - 1):
            s += ", "

    if len(inputs):
        s += ", "

    # output
    s += "&(" + prefix + node["name"] + ")"

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

        s = "    std::vector<float> {};\n".format(node["name"])

        ret += s

    ret += '''
};
'''

    return ret


def EmitInit(deque_graph, # type: collections.deque
            ctx, # type: Context
            ):

    """
    Emit initialization code(const, reshape)
    """

    ret = ""

    # header
    ret = '''
void NetworkInit(Buffer *buffer) {
'''

    for layer_name in deque_graph:
        node = ctx.graph[layer_name]

        if node["op"] == "NoOp":
            continue


        if node["op"] == "Const":

            # Emit initializer function.
            s = '''
    ConstInit_$name(${args});
'''
            st = Template(s)
            d = node
            d["args"] = ConstructArgString("buffer->", node)

            ret += st.substitute(d)

        elif node["op"] == "Placeholder":

            # Emit initializer function.
            s = '''
    PlaceholderInit_$name(${args});
'''
            st = Template(s)
            d = node
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

// Initialize buffer for the network
void NetworkInit(Buffer *buffer);

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

    # Emit initializer function
    str_cc += EmitInit(deq, ctx)

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
