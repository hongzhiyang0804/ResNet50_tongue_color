import os
import tensorflow as tf
import dataset
from tensorflow.contrib.slim import nets

slim = tf.contrib.slim
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batch_size = 16
num_classes = 5
total_epoch = 60

inputs = tf.placeholder(tf.float32, shape=[None, 416, 416, 3], name='inputs')
labels = tf.placeholder(tf.int32, shape=[None, num_classes], name='labels')
is_training = tf.placeholder(tf.bool, name='is_training')

net, endpoints = nets.resnet_v2.resnet_v2_50(inputs, num_classes=num_classes,
                                             is_training=is_training)
net = tf.squeeze(net, axis=[1, 2])

# 损失函数
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=labels, logits=net))
# softmax分类器,计算概率
logits = tf.nn.softmax(net, name='softmax')

# 预测精度
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 优化器
optimizer = tf.train.AdamOptimizer(1e-4)
train = optimizer.minimize(cost)

init = tf.global_variables_initializer()

saver = tf.train.Saver(tf.global_variables(), max_to_keep=15)

# 获取图像数据
train_image_list, train_label_list = dataset.load_satetile_image('./data/train/', dataset='train')
valid_image_list, valid_label_list = dataset.load_satetile_image('./data/valid/', dataset='valid')

def Evaluate(sess):
    test_acc = 0.0
    test_loss = 0.0
    for it in range(len(valid_image_list) // batch_size):
        valid_img, valid_label = dataset.batch_image(valid_image_list, valid_label_list, batch_size=batch_size, index=it)
        test_feed_dict = {inputs: valid_img,
                          labels: valid_label,
                          is_training: False}
        _, loss_ = sess.run([train, cost], feed_dict=test_feed_dict)
        acc_ = accuracy.eval(feed_dict=test_feed_dict)
        test_loss += loss_
        test_acc += acc_
        if it == ((len(valid_image_list) // batch_size) - 1):
            test_loss /= (len(valid_image_list) // batch_size)
            test_acc /= (len(valid_image_list) // batch_size)
    summary = tf.Summary(value=[tf.Summary.Value(tag='test_loss', simple_value=test_loss),
                                tf.Summary.Value(tag='test_accuracy', simple_value=test_acc)])

    return test_acc, test_loss, summary


with tf.Session() as sess:
    sess.run(init)
    # tensorboard生成
    summary_writer = tf.summary.FileWriter('./logs', sess.graph)
    # 每个epoch训练
    for epoch in range(1, total_epoch + 1):
        train_acc = 0.0
        train_loss = 0.0
        iteration = 20
        for step in range(len(train_image_list) // batch_size):
            # 加载训练集和验证集
            img, img_label = dataset.batch_image(train_image_list, train_label_list, batch_size=batch_size, index=step)
            # print(img.shape, img_label.shape)
            train_feed_dict = {inputs: img,
                               labels: img_label,
                               is_training: True}
            _, batch_loss = sess.run([train, cost], feed_dict=train_feed_dict)
            batch_acc = accuracy.eval(feed_dict=train_feed_dict)
            train_loss += batch_loss
            train_acc += batch_acc
            if step % 20 == 0:
                train_loss /= (step + 1)  # average loss
                print('epoch: %d, iteration: %d, train_loss: %.4f' % (epoch, iteration, train_loss))
                iteration += 20
            if step == ((len(train_image_list) // batch_size) - 1):
                train_loss /= (len(train_image_list) // batch_size)  # average accuracy
                train_acc /= (len(train_image_list) // batch_size)  # average accuracy
                train_summary = tf.Summary(value=[tf.Summary.Value(tag='train_loss', simple_value=train_loss),
                                                  tf.Summary.Value(tag='train_accuracy', simple_value=train_acc)])
                test_acc, test_loss, test_summary = Evaluate(sess)
                #
                summary_writer.add_summary(summary=train_summary, global_step=epoch)
                summary_writer.add_summary(summary=test_summary, global_step=epoch)
                summary_writer.flush()

                line = "epoch: %d/%d, train_loss: %.4f, train_acc: %.4f, test_loss: %.4f, test_acc: %.4f \n" % (
                    epoch, total_epoch, train_loss, train_acc, test_loss, test_acc)
                print(line)

                with open('logs.txt', 'a') as f:
                    f.write(line)
        saver.save(sess=sess, save_path='./ResNet_50.ckpt', global_step=epoch)















