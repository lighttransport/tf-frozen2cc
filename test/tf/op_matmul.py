import tensorflow as tf

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

if __name__ == '__main__':
    # Assume 1-D vector
    input_shape = [1, 8]
    test_matmul(input_shape)

