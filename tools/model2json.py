#!/usr/bin/env python

import os, sys
import json

# Simple trained model to JSON converter.
# Reference
# https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

def proc(model_dir, output_node_name):

    import tensorflow as tf

    from google.protobuf import text_format
    from tensorflow.core.framework import graph_pb2
    from google.protobuf.json_format import MessageToJson

    # We retrieve our checkpoint fullpath
    checkpoint = tf.train.get_checkpoint_state(model_dir)
    input_checkpoint = checkpoint.model_checkpoint_path
    
    # We precise the file fullname of our freezed graph
    absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
    #output_graph = absolute_model_dir + "/frozen_model.pb"

    # We clear devices to allow TensorFlow to control on which device it will load operations
    clear_devices = True

    with tf.Session(graph=tf.Graph()) as sess:
        saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

        saver.restore(sess, input_checkpoint)

        # assume single output node name.
        output_graph_def = tf.graph_util.convert_variables_to_constants(sess, tf.get_default_graph().as_graph_def(), [output_node_name])

        # remove training nodes
        inference_graph = tf.graph_util.remove_training_nodes(output_graph_def)
    
        # to JSON
        j = MessageToJson(inference_graph)

        with open("output.json", "w") as o:
            o.write(j)

def main():

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory")
    #parser.add_argument("--input_text", action='store_true', default=False, help="Input protobuf file is text")
    parser.add_argument("--output_node_name", type=str, required=True, help="Output node name")
    args = parser.parse_args()

    proc(args.model_dir, args.output_node_name)
    

main()
