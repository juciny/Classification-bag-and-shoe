import os
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from PIL import Image
img=Image.open('F:/jupyterNotebook/MyNet001/taobao_photos/bag/8426.jpg')
img.show()

'''

plt.plot([10, 20, 30])
plt.xlabel('tiems')
plt.ylabel('numbers')
plt.show()
'''



# 比较resize_image_with_crop_or_pad和resize_images
#image=cv2.imread('F:/jupyterNotebook/MyNet001/taobao_photos/bag/8426.jpg')
#image2=tf.image.resize_image_with_crop_or_pad(image,64,64)
#image3=tf.image.resize_images(image,[120,120],method=1)

#with tf.Session() as sess:
#sess.run(tf.global_variables_initializer())
# print(image.shape)
# cv2.imshow("原图",image)
# #image2=sess.run(image2)
#  # cv2.imshow("image2",image2)
#  image3=sess.run(image3)
#  #cv2.imshow("image3",image3)
#   cv2.waitKey(0)
