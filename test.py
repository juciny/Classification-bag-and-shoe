from PIL import Image
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import model
from input_data import get_files
import os

#从指定目录中选取一张图片
def get_one_image(train):
    files = os.listdir(train)
    n = len(files)
    ind = np.random.randint(0,n)
    img_dir = os.path.join(train,files[ind])
    image = Image.open(img_dir)
    plt.imshow(image)
    plt.show()
    image = image.resize([64, 64])
    image = np.array(image)
    return image

def evaluate_one_image():

    train='F:/jupyterNotebook/MyNet001/taobao_photos/test'
    image_array=get_one_image(train)
    print(image_array)

    with tf.Graph().as_default():
        BATCH_SIZE=1
        N_CLASSES=2

        image=tf.cast(image_array,tf.float32)

        image=tf.image.per_image_standardization(image)

        image=tf.reshape(image,[1,64,64,3])

        logit = model.inference(image, BATCH_SIZE, N_CLASSES)
        logit = tf.nn.softmax(logit)

        x = tf.placeholder(tf.float32, shape=[64, 64, 3])
        logs_train_dir = 'F:/jupyterNotebook/MyNet001/log/'

        #saver=tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            print("从指定的路径中加载模型。。。。")

            ckpt=tf.train.get_checkpoint_state(logs_train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                #saver.restore(sess, ckpt.model_checkpoint_path)
                print('模型加载成功, 训练的步数为 %s' % global_step)

            else:
                print('模型加载失败，，，文件没有找到')

                # 将图片输入到模型计算
            prediction = sess.run(logit, feed_dict={x: image_array})
                # 获取输出结果中最大概率的索引
            max_index = np.argmax(prediction)
            if max_index == 0:
                print('包包的概率 %.6f' % prediction[:, 0])
            else:
                print('鞋子的概率 %.6f' % prediction[:, 1])



evaluate_one_image()




