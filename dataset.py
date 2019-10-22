import cv2
import numpy as np
import os
from keras.utils import to_categorical
from PIL import Image, ImageFilter
import random
# 载入数据路径
def load_satetile_image(input_dir, dataset='train'):
    label_list = []
    image_path = []
    # train处理
    if dataset == 'train':
        path = input_dir
        for child_dir in os.listdir(path):
            child_path = os.path.join(path, child_dir)
            last_name = child_path.split('/')
            child_path_last = last_name[-1]
            # print(child_path_last)
            # 获取各类别图片路径并载入矩阵,同时赋予相应的标签
            if child_path_last == 'lightred':
                dir_counter = 0
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'cyan':
                dir_counter = 1
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'red':
                dir_counter = 2
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'lightwhite':
                dir_counter = 3
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'dark':
                dir_counter = 4
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
        # 随机打散不同类别图片路径及对应标签
        cc = list(zip(image_path, label_list))
        random.shuffle(cc)
        image_path[:], label_list[:] = zip(*cc)
    # valid处理
    else:
        path = input_dir
        for child_dir in os.listdir(path):
            child_path = os.path.join(path, child_dir)
            last_name = child_path.split('/')
            child_path_last = last_name[-1]
            # print(child_path_last)
            # 获取各类别图片路径并载入矩阵,同时赋予相应的标签
            if child_path_last == 'lightred':
                dir_counter = 0
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'cyan':
                dir_counter = 1
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'red':
                dir_counter = 2
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'lightwhite':
                dir_counter = 3
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
            elif child_path_last == 'dark':
                dir_counter = 4
                for dir_image in os.listdir(path + child_path_last):
                    image_path.append(os.path.join(child_path, dir_image))
                    label_list.append(dir_counter)
        # 随机打散不同类别图片路径及对应标签
        cc = list(zip(image_path, label_list))
        random.shuffle(cc)
        image_path[:], label_list[:] = zip(*cc)
    return image_path, label_list
# 批量读取图片及对应标签
def batch_image(image_path, label_list, batch_size=64,index=0):
    img_list = []
    # 获取每批次下图片以及对应标签
    for j in image_path[index*batch_size: (index+1)*batch_size]:
        image = Image.open(j)
        # 图片缩放
        image = image.resize((416, 416))
        # 图片模糊
        blurry_image = image.filter(ImageFilter.BLUR)
        # 图片中间位置置零
        zero_img = np.array(blurry_image)
        weight = zero_img.shape[0]
        height = zero_img.shape[1]
        row_first = int(0.2 * weight)
        row_final = int(0.8 * weight)
        column_first = int(0.2 * height)
        column_final = int(0.8 * height)
        zero_img[row_first:row_final, column_first:column_final, :] = 0
        # 将rgb转为bgr,便于cv2转换
        img_bgr = cv2.cvtColor(zero_img, cv2.COLOR_RGB2BGR)
        # 进行颜色空间转换为HSV
        HSV = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        # 将bgr转为rgb
        final_img = cv2.cvtColor(HSV, cv2.COLOR_BGR2RGB)
        # 转为numpy数组形式
        img = np.array(final_img)
        # print(img)
        img = img / 255.0
        img_list.append(img)
    # 得到每批次的标签
    batch_size_label_list = label_list[index * batch_size:(index+1) * batch_size]
    # print(img_list)
    # 得到每批次图片
    x_batch = np.array(img_list)
    # print(x_batch.shape)
    # 标签转为one-hot
    y_batch = to_categorical(batch_size_label_list, 5)
    # print(y_batch)
    return x_batch, y_batch

# image_path, label_list = load_satetile_image('./data/train/', dataset='train')
# print(len(image_path),len(label_list))
# for i in range(10):
#     batch_image(image_path, label_list, batch_size=8, index=i)



