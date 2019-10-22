import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np
import cv2
import os
import time
from tensorflow.contrib.slim import nets
# 超参数设置
training_flag = tf.cast(False, tf.bool)

# 模型保存的路径和文件名
model_path = './model/ResNet_50.ckpt-30'
saver = tf.train.import_meta_graph("./model/ResNet_50.ckpt-30.meta", clear_devices=True)
# sess = tf.Session()
# # 加载模型和训练好的参数
# saver.restore(sess, model_path)
tongue_detection_image = './data/valid/cyan/195412_9b0bf021a9c7428e929b7a602d74a6ee.png'
tongue_color_image_size = 416
num_classes = 5
is_training = tf.cast(False, tf.bool)
tongue_color_save_path = './data/0.png'
# 舌色图像预处理
image1 = Image.open(tongue_detection_image)
shrink_image = image1.resize((tongue_color_image_size, tongue_color_image_size))
blurry_image = shrink_image.filter(ImageFilter.BLUR)
zero_img = np.array(blurry_image)
weight = zero_img.shape[0]
height = zero_img.shape[1]
row_first = int(0.2 * weight)
row_final = int(0.8 * weight)
column_first = int(0.2 * height)
column_final = int(0.8 * height)
zero_img[row_first:row_final, column_first:column_final, :] = 0
img_bgr = cv2.cvtColor(zero_img, cv2.COLOR_RGB2BGR)
HSV = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
HSV = cv2.cvtColor(HSV, cv2.COLOR_BGR2RGB)
final_img = Image.fromarray(HSV.astype('uint8')).convert('RGB')
final_img.save(tongue_color_save_path)

# 加载需要预测的图片
image_data = tf.gfile.FastGFile(tongue_color_save_path, 'rb').read()

# 将图片格式转换成我们所需要的矩阵格式，第二个参数为3，代表3维
decode_image = tf.image.decode_png(image_data, 3)

# 再把数据格式转换成能运算的float32
decode_image = tf.image.convert_image_dtype(decode_image, tf.float32)

# 转换成指定的输入格式形状
image = tf.reshape(decode_image, [-1, tongue_color_image_size, tongue_color_image_size, 3])
# 定义预测结果为logit值最大的分类，这里是前向传播算法，也就是卷积层、池化层、全连接层那部分

net, endpoints = nets.resnet_v2.resnet_v2_50(image, num_classes=num_classes,
                                                 is_training=is_training)
net = tf.squeeze(net, axis=[1, 2])
# 利用softmax来获取概率
probabilities = tf.nn.softmax(net)

# 获取最大概率的标签位置
correct_prediction = tf.argmax(net, 1)

# 定义Savar类
# saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, model_path)
    # 获取预测结果
    probabilities, labels = sess.run([probabilities, correct_prediction])
    # print(probabilities)
    print(probabilities[0][labels], labels)
    if labels == 0:
        tongue_color_label = 'lightred'
        print(tongue_color_label, probabilities[0][labels])
    elif labels == 1:
        tongue_color_label = 'cyan'
        print(tongue_color_label, probabilities[0][labels])
    elif labels == 2:
        tongue_color_label = 'red'
        print(tongue_color_label, probabilities[0][labels])
    elif labels == 3:
        tongue_color_label = 'lightwhite'
        print(tongue_color_label, probabilities[0][labels])
    elif labels == 4:
        tongue_color_label = 'dark'
        print(tongue_color_label, probabilities[0][labels])
    os.remove(tongue_color_save_path)

