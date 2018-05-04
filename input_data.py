import math
import tensorflow as tf
import os
import numpy as np

bag=[]
label_bag=[]
shoe=[]
label_shoe=[]


def get_files(file_dir,ratio):

    for file in os.listdir(file_dir + '/bag'):
        bag.append(file_dir + '/bag/' + file)
        label_bag.append(0)
    for file in os.listdir(file_dir + '/shoe'):
        bag.append(file_dir + '/shoe/' + file)
        label_bag.append(1)

    # 把所有的image都合成一行数组
    image_list = np.hstack((bag, shoe))
    # 把所有的label都合成一行数组
    label_list = np.hstack((label_bag, label_shoe))

    # 把image和label合成一个二维数组，两行
    temp = np.array([image_list, label_list])
    # 把上面的二维数组转置，得到一个两列的数组
    temp = temp.transpose()
    # 将数字顺序打乱
    np.random.shuffle(temp)

    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # 通过一个比率ratio获得train和val的大小
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))
    n_train = n_sample - n_val

    train_image = all_image_list[0:n_train]
    train_label = all_label_list[0:n_train]
    train_label = [int(float(i)) for i in train_label]
    val_image = all_image_list[n_train:-1]
    val_label = all_label_list[n_train:-1]
    val_label = [int(float(i)) for i in val_label]

    return train_image, train_label, val_image, val_label


# 生成batch

def get_batch(image, label, image_w, image_h, batch_size, capacity):

    # 将image和label转换成相应的格式
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)

    input_queue = tf.train.slice_input_producer([image, label])
    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # 将图像解码
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # 对数据做预处理
    # image = tf.image.resize_image_with_crop_or_pad(image, image_w, image_h)
    image = tf.image.resize_images(image, [image_w, image_h], method=1)

    image = tf.image.per_image_standardization(image)

    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)

    image_batch = tf.cast(image_batch, tf.float32)

    label_batch = tf.reshape(label_batch, [batch_size])

    return image_batch,label_batch