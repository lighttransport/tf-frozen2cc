import sys
import os
import json

import tensorflow as tf
from google.protobuf import json_format
from tensorflow.core.framework import graph_pb2

if __name__ == '__main__':
       
    with open(sys.argv[1], "r") as f:
        graph_def = json_format.Parse(f.read(), graph_pb2.GraphDef())

        with tf.Session() as sess:

            c = tf.constant([1.0, 2.0, 3.0])
            print(type(c))
            
            tf.import_graph_def(graph_def)

            d = tf.get_default_graph().get_tensor_by_name("import/const1:0")
            print("d", d.eval())
            a = tf.add(d, d)
            print("a", a.eval())

            #sess.run(a)
            #print(a.eval())

            #print(tf.get_default_graph().collections)
            #print(tf.get_default_graph().as_graph_def().node)
            #print(tf.get_default_graph().as_graph_def().node)

        #for node in msg.node:
        #    print(node.attr.value)

        #with tf.Graph().as_default() as graph:
        #    tf.import_graph_def(msg)
        # 
        #    print(graph)

        #for op in graph:
        #    print(op)


