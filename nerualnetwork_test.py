'''
《TensorFlow入门与实战》 P78
训练神经网络拟合正弦函数的代码示例
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pylab


# 绘制标准y = sin(x)曲线
def draw_correct_line():
    x = np.arange(0, 2 * np.pi, 0.01)
    # 转换为len(x) * 1矩阵
    x = x.reshape(len(x), 1)
    y = np.sin(x)

    pylab.plot(x, y, label="标准sin曲线")
    # 坐标轴为红色
    plt.axhline(linewidth=1, color='r')


def get_train_data():
    '''
    返回一个训练样本(train_x, train_y)
    其中，train_x是随机的自变量，train_y是train_x的sin函数值
    '''
    train_x = np.random.uniform(0.0, 2 * np.pi, (1))
    train_y = np.sin(train_x)
    return train_x, train_y


def inference(input_data):
    '''
    定义前向计算的网络结构
    :param input_data:
    :return:output_data
    '''
    # 第一个隐层，16个节点
    with tf.variable_scope("hidden1"):
        # weights、biases、hidden1都是在hidden1作用域下的共享变量，初始化只在第一次迭代时起作用，下同
        # 因此，每次迭代（前向计算）时，相应的值都会更新
        weights = tf.get_variable("weight", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("biases", [1, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        hidden1 = tf.sigmoid(tf.multiply(input_data, weights) + biases)

    # 第二个隐层，16个节点
    with tf.variable_scope("hidden2"):
        weights = tf.get_variable("weight", [16, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("biases", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        mul = tf.matmul(hidden1, weights)
        hidden2 = tf.sigmoid(mul + biases)

    # 第三个隐层，16个节点
    with tf.variable_scope("hidden3"):
        weights = tf.get_variable("weight", [16, 16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("biases", [16], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        mul = tf.matmul(hidden2, weights)
        hidden3 = tf.sigmoid(mul + biases)

    # 输出层
    with tf.variable_scope("output_layer"):
        weights = tf.get_variable("weights", [16, 1], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        biases = tf.get_variable("biases", [1], tf.float32, initializer=tf.random_normal_initializer(0.0, 1))
        output_data = tf.matmul(hidden3, weights) + biases

    return output_data


def train():
    # 学习率
    learning_rate = 0.01

    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)

    # 网络的实际输出
    net_out = inference(x)

    # 损失函数
    loss_op = tf.square(net_out - y)

    # 采用随机梯度下降的优化函数（误差反向传播）
    opt = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = opt.minimize(loss_op)

    # 初始化所有变量
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print("开始训练")
        # 迭代一百万次
        for i in range(1000000):
            # 获得训练样本
            train_x, train_y = get_train_data()
            # train_op中需要用到x和y
            # 每run一次，相当于训练（迭代）了一次
            sess.run(train_op, feed_dict={x: train_x, y: train_y})

            # 每训练一万次，绘制一次对比图
            if i % 10000 == 0:
                # times表示“第times万次”
                times = int(i / 10000)
                print("第", times, "万次")
                test_x_ndarray = np.arange(0, 2 * np.pi, 0.01)
                test_y_ndarray = np.zeros([len(test_x_ndarray)])
                ind = 0

                for test_x in test_x_ndarray:
                    # net_out中需要用到x
                    # net_out的输出是目前网络拟合的结果，赋给test_y
                    test_y = sess.run(net_out, feed_dict={x: test_x})
                    # test_y_ndarray向量中的第ind个元素 被 test_y中的第ind个元素代替
                    np.put(test_y_ndarray, ind, test_y)
                    ind = ind + 1

                # 先绘制标准的y = sin(x)曲线
                draw_correct_line()
                # 再绘制目前拟合的曲线，以虚线表示
                pylab.plot(test_x_ndarray, test_y_ndarray, '--', label="第" + str(times) + "万次")
                pylab.show()


if __name__ == "__main__":
    train()
