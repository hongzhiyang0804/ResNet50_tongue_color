import tensorflow as tf
from PIL import Image, ImageFilter
import numpy as np
import cv2
import time
import os
# 超参数设置
training_flag = tf.cast(False, tf.bool)

# 模型保存的路径和文件名
model_path = './model/ResNet_50.ckpt-30'
# 启动会话,载入网络结构,固定网络
sess = tf.Session()
# 直接通过训练参数名称就可以获取需要的参数
saver = tf.train.import_meta_graph("./model/ResNet_50.ckpt-30.meta", clear_devices=True)
# 加载模型和训练好的参数
saver.restore(sess, model_path)
# tongue_detction_image = './data/valid/cyan/190648_30fc383127a74113a1917020230a8661.png'
tongue_color_save_path = './data/0.png'
tongue_color_image_size = 416
num_classes = 5

def tongue_color(tongue_detction_image):
    ret = {}
    # 舌色图像预处理
    image1 = Image.open(tongue_detction_image)
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
    # 获取placeholder变量
    input_x = sess.graph.get_tensor_by_name('inputs:0')
    flag = sess.graph.get_tensor_by_name('is_training:0')
    # 获取需要进行计算的operator
    op = sess.graph.get_tensor_by_name('softmax:0')
    # 加载需要预测的图片
    image_data = tf.gfile.FastGFile(tongue_color_save_path, 'rb').read()
    # 将图片格式转换成我们所需要的矩阵格式，第二个参数为3，代表3维
    decode_image = tf.image.decode_png(image_data, 3)
    # 再把数据格式转换成能运算的float32
    decode_image = tf.image.convert_image_dtype(decode_image, tf.float32)
    # 转换成指定的输入格式形状
    image = tf.reshape(decode_image, [-1, tongue_color_image_size, tongue_color_image_size, 3])
    image_tensor = sess.run(image)
    flag_tensor = sess.run(training_flag)
    tongue_color_predictions = sess.run(op, {input_x: image_tensor, flag: flag_tensor})
    labels = np.argmax(tongue_color_predictions, 1)
    # print(labels)
    print(tongue_color_predictions[0][labels])
    # 输出判别结果
    if labels == 0:
        tongue_color_label = '淡红'
        ret['舌色'] = tongue_color_label
        # print(tongue_color_label, tongue_color_predictions[0][labels])
    elif labels == 1:
        tongue_color_label = '青紫'
        # print(tongue_color_label, tongue_color_predictions[0][labels])
        ret['舌色'] = tongue_color_label
    elif labels == 2:
        tongue_color_label = '红'
        # print(tongue_color_label, tongue_color_predictions[0][labels])
        ret['舌色'] = tongue_color_label
    elif labels == 3:
        tongue_color_label = '淡白'
        # print(tongue_color_label, tongue_color_predictions[0][labels])
        ret['舌色'] = tongue_color_label
    elif labels == 4:
        tongue_color_label = '暗'
        # print(tongue_color_label, tongue_color_predictions[0][labels])
        ret['舌色'] = tongue_color_label
    # print(ret)
    os.remove(tongue_color_save_path)
    return ret
if __name__ == "__main__":
    # tongue_detection_image = './tongue_docking/123.jpg'
    t1 = time.time()
    print(tongue_color('./data/valid/lightred/072601_5fbe63cbaffa44d8a4b5c4be7bc565a8.png'))
    t2 = time.time()
    print('t1=', t2-t1)
    print(tongue_color('./data/valid/dark/094922_0c150f0e401046a498139e352fc6ede7.png'))
    print('t2=', time.time()-t2)
