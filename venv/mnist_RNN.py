from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def compute_accuracy(v_x, v_y):
    global prediction
    y_pre = sess.run(prediction, feed_dict={x: v_x, keep_prob: 1})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    result = sess.run(accuracy, feed_dict={x: v_x, y: v_y, keep_prob: 1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

def conv_pool_layer(X, img_len, img_hi, out_seq):
    W = weight_variable([img_len, img_len, img_hi, out_seq])
    b = bias_variable([out_seq])
    h_conv = tf.nn.relu(conv2d(X, W) + b)
    return max_pool_2x2(h_conv)

def lstm(X):
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(256, forget_bias=1.0, state_is_tuple=True)
    _init_state = lstm_cell.zero_state(128, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X, initial_state=_init_state, time_major=False)
    W = weight_variable([256, 8])
    b = bias_variable([8])
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    results = tf.matmul(outputs[-1], W) + b
    return results

# load mnist data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)

# reshape image
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_pool1 = conv_pool_layer(x_image, 5, 1, 32)
h_pool2 = conv_pool_layer(h_pool1, 5, 32, 64)

# reshape data
X_in = tf.reshape(h_pool2, [-1, 49, 64])
X_in = tf.transpose(X_in, [0, 2, 1])

# put into a lstm layer
prediction = lstm(X_in)

# calculate the loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

# use Gradient descent optimizer(梯度下降法)
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
# init session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(10000):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    sess.run(train_step, feed_dict={x: batch_x, y: batch_y, keep_prob: 0.5})
    if i % 50 == 0:
        print("acc:%f"%sess.run(accuracy, feed_dict={x: batch_x, y: batch_y, }))
