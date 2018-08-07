import tensorflow as tf

from google.protobuf.json_format import MessageToJson

from tensorflow.core.framework import graph_pb2

def test_matmul(input_shape):

    a_rnd = tf.random_uniform(input_shape, dtype=tf.float32, seed=None)
    b_rnd = tf.matrix_transpose(tf.random_uniform(input_shape, dtype=tf.float32, seed=None))

    with tf.Session() as sess:

        # TODO(syoyo): transpose, adjoint
        op = tf.matmul(a_rnd, b_rnd)

        ret = sess.run(op)
        print('a', a_rnd.eval())
        print('b', b_rnd.eval())
        print('result', ret)

	graph_def = sess.graph_def

	j = MessageToJson(graph_def)
		
	with open("output.json", "w") as o:
	    o.write(j)

if __name__ == '__main__':
    # Assume 1-D vector
    input_shape = [1, 8]
    test_matmul(input_shape)

