
import tensorflow as tf

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1, dtype=tf.float32)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(value=0.1, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)


def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 1, 1], strides=[1, 2, 2, 1], padding='SAME')


def inference(images, batch_size, n_classes):
    w_conv1 = weight_variable([3, 3, 3, 64])
    b_conv1 = bias_variable([64])
    h_conv1 = tf.nn.relu(conv2d(images, w_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    w_conv2 = weight_variable([3, 3, 64, 16])
    b_conv2 = bias_variable([16])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    h_pool2_flat = tf.reshape(h_pool2, shape=[batch_size, -1])
    dim = h_pool2_flat.get_shape()[1].value
    w_fc1 = weight_variable([dim, 128])
    b_fc1 = weight_variable([128])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    # keep_prob = tf.placeholder(tf.float32)
    # h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    w_fc2 = weight_variable([128, n_classes])
    b_fc2 = weight_variable([n_classes])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, w_fc2) + b_fc2)

    return y_conv


# 定义损失函数
# 传入参数：logits，网络计算输出值。labels，真实值，在这里是0或者1
def losses(logits, labels):
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels)
    loss = tf.reduce_mean(cross_entropy, name='loss')

    return loss


# 获取训练步长
def trainning(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    return train_step


# 定义准确率
def evaluation(logits, labels):
    correct_prediction= tf.nn.in_top_k(logits,labels,1)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    return accuracy