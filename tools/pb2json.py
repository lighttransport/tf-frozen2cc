#!/usr/bin/env python

import os, sys
import json

# Simple .pb binary to JSON converter.

def proc(input_filename, input_is_text, output_filename):

    print("Input", input_filename)
    print("Input is text", input_is_text)
    print("Output", output_filename)

    import tensorflow as tf

    from google.protobuf import text_format
    from tensorflow.core.framework import graph_pb2
    from google.protobuf.json_format import MessageToJson

    with open(input_filename, "rb") as f:
        graph_def = graph_pb2.GraphDef()
        if input_is_text == True:
            text_format.Merge(f.read(), graph_def)
        else:
            graph_def.ParseFromString(f.read())
        
        j = MessageToJson(graph_def)

        with open(output_filename, "w") as o:
            o.write(j)

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input model in JSON")
    parser.add_argument("--input_text", action='store_true', default=False, help="Input protobuf file is text")
    parser.add_argument("--output", type=str, required=True, help="Output C++ base filename")
    args = parser.parse_args()

    proc(args.input, args.input_text, args.output)
    

main()
