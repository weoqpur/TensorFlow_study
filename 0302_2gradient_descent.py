import tensorflow as tf
import numpy as np

# 랜덤 시드 초기화 다음에도 실행 했을때 똑같은 값이 나오게
tf.random.set_seed(0)

x_data = [1., 2., 3., 4.]
y_data = [1., 3., 5., 7.]

# W = tf.Variable(tf.random.normal([1], -100., 100.))
W = tf.Variable([5.0])

for step in range(300):
    hypothesis = W * x_data
    cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    alpha = 0.01
    # multiply : 두 수를 곱해 줌
    gradient = tf.reduce_mean(tf.multiply(tf.multiply(W, x_data) - y_data, x_data))
    descent = W - tf.multiply(alpha, gradient)
    W.assign(descent)

    if step % 10 == 0:
        print("{:5} | {:10.4f} | {:10.6f}".format(step, cost.numpy(), W.numpy()[0]))


