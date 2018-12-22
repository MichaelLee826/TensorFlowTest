import tensorflow as tf
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.Session() as sess:
    identity_matrix = tf.diag([1.0, 1.0, 1.0])
    A = tf.truncated_normal([2, 3])
    B = tf.fill([2, 3], 5.0)
    C = tf.random_uniform([3, 2], minval=1, maxval=10)
    D = tf.convert_to_tensor(np.array([[1., 2., 3.], [-3., -7., -1.], [0., 5., -2.]]))

    print('A =', sess.run(A))
    print('B =', sess.run(B))
    print('C =', sess.run(C))
    print('D =', sess.run(D))

    # 矩阵乘法
    tf.matmul(B, identity_matrix)

    # 转置
    tf.transpose(C)
    # print(sess.run(tf.transpose(C)))

    # 求行列式
    tf.matrix_determinant(D)
    # print(sess.run(tf.matrix_determinant(D)))

    # 求逆
    tf.matrix_inverse(D)

    # 求特征值和特征向量
    # 输出结果中，第一个array为特征值，另一个array是对应的特征向量
    tf.self_adjoint_eig(D)
    #print(sess.run(tf.self_adjoint_eig(D)))
