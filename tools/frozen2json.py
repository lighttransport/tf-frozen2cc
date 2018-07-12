#!/usr/bin/env python

import os, sys
import json

# Simple freezed graph .pb to JSON converter.

def proc(input_filename, input_text, output_filename):

    import tensorflow as tf

    from google.protobuf import text_format
    from tensorflow.core.framework import graph_pb2
    from google.protobuf.json_format import MessageToJson

    graph_def = graph_pb2.GraphDef()

    with open(input_filename, "rb") as f:
        if input_text:
            text_format.Merge(f.read(), graph_def)
        else:
            graph_def.ParseFromString(f.read())
    
    # remove training nodes
    inference_graph = tf.graph_util.remove_training_nodes(graph_def)
    
    # to JSON
    j = MessageToJson(inference_graph)

    with open(output_filename, "w") as o:
        o.write(j)

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Freezed graph .pb")
    parser.add_argument("--input_text", action='store_true', default=False, help="Input protobuf file is text")
    parser.add_argument("--output", type=str, required=True, help="Output JSON filename")
    args = parser.parse_args()

    proc(args.input, args.input_text, args.output)
    

main()
