'''
《TensorFlow入门与实战》 P51
通过tensorboard查看数据变化曲线的代码示例
'''

import numpy as np
import tensorflow as tf

x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

# 权重
weight = tf.get_variable("weight", [], tf.float32, initializer=tf.random_normal_initializer())
# 偏置
bias = tf.get_variable("bias", [], tf.float32, initializer=tf.random_normal_initializer())
# 目标函数： Y = 2 * X + b = weight * x + bias
pred = tf.add(tf.multiply(x, weight, name="mul_op"), bias, name="add_op")
# 损失函数： 平方差
loss = tf.square(y - pred, name="loss")

# 优化函数
optimizer = tf.train.GradientDescentOptimizer(0.01)
# 计算梯度
grads_and_vars = optimizer.compute_gradients(loss)
# 应用梯度
train_op = optimizer.apply_gradients(grads_and_vars)

# 收集训练过程中weight、bias、loss的值
tf.summary.scalar("weight", weight)
tf.summary.scalar("bias", bias)
tf.summary.scalar("loss", loss[0])
# 将以上所有的summary合并到一起
merged_summary = tf.summary.merge_all()

# 文件路径：E:\PyCharm-Workspace\TensorFlowTest\log_graph
summary_writer = tf.summary.FileWriter('./log_graph')
summary_writer.add_graph(tf.get_default_graph())

# 所有变量初始化
init_op = tf.global_variables_initializer()

# 在Session中进行训练
with tf.Session() as sess:
    sess.run(init_op)
    # 训练500次
    for step in range(500):
        train_x = np.random.randn(1)
        train_y = 2 * train_x + np.random.randn(1) * 0.01 + 10
        _, summary = sess.run([train_op, merged_summary], feed_dict={x:train_x, y:train_y})
        summary_writer.add_summary(summary, step)

# 运行成功后，在cmd中输入：tensorboard --logdir=./log_graph，然后在浏览器(127.0.0.1:6006)中可查看生成的图
