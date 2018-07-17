import tensorflow as tf

def test_conv2d(input_shape, filter_shape, strides, padding):

    input_rnd = tf.random_uniform(input_shape, dtype=tf.float32, seed=None)
    filter_rnd = tf.random_uniform(filter_shape, dtype=tf.float32, seed=None)

    with tf.Session() as sess:
        strides = [1, 1, 1, 1]
        padding = 'SAME'

        op = tf.nn.conv2d(input=input_rnd, filter=filter_rnd, strides=strides, padding=padding)

        ret = sess.run(op)
        print('input', input_rnd.eval())
        print('filter', filter_rnd.eval())
        print('result', ret)

if __name__ == '__main__':
    input_shape = [1, 8, 8, 1]
    filter_shape = [3, 3, 1, 1]
    strides = [1, 1, 1, 1]
    padding = 'SAME'
    test_conv2d(input_shape, filter_shape, strides, padding)

