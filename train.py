import os
import numpy as np
import tensorflow as tf
import input_data
import model


# 变量声明
N_CLASSES = 4
IMG_W = 64   # resize图像，太大的话训练时间久
IMG_H = 64
BATCH_SIZE =20
CAPACITY = 200
MAX_STEP = 200     # 一般大于10K
learning_rate = 0.0001   # 学习速度一般小于0.0001

train_dir='F:/jupyterNotebook/MyNet001/taobao_photos'
logs_train_dir='F:/jupyterNotebook/MyNet001/log'

train, train_label, val, val_label = input_data.get_files(train_dir, 0.3)

train_batch,train_label_batch = input_data.get_batch(train, train_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

val_batch, val_label_batch = input_data.get_batch(val, val_label, IMG_W, IMG_H, BATCH_SIZE, CAPACITY)

# 获取估计的标签
train_logits = model.inference(train_batch, BATCH_SIZE, N_CLASSES)

# 获取loss
train_loss = model.losses(train_logits, train_label_batch)

# 获取学习速率
train_op = model.trainning(train_loss, learning_rate)

# 获取准确率
train_acc = model.evaluation(train_logits, train_label_batch)

test_logits = model.inference(val_batch, BATCH_SIZE, N_CLASSES)
test_loss = model.losses(test_logits, val_label_batch)
test_acc = model.evaluation(test_logits, val_label_batch)


sess=tf.Session()

# 产生一个saver来存储训练好的模型
saver = tf.train.Saver()

sess.run(tf.global_variables_initializer())

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess,coord=coord)

try:

    for step in np.arange(MAX_STEP):
        if coord.should_stop():
            break
        _, tra_loss, tra_acc = sess.run([train_op, train_loss, train_acc])
        if step % 10 == 0:
            print('Step %d, train loss = %.2f, train accuracy = %.2f%%' %(step, tra_loss, tra_acc*100.0))
        if (step+1)==MAX_STEP:
            checkpoint_path=os.path.join(logs_train_dir,'model.ckpt')
            saver.save(sess,checkpoint_path,global_step=step)

except tf.errors.OutOfRangeError:
    print('Done training--epoch limit reached')

finally:
    coord.request_stop()


