import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

# 인풋 데이터
x_data = [1, 2, 3, 4, 5] 
# 아웃풋 데이터
y_data = [1, 2, 3, 4, 5]

# 원래는 랜덤 값을 지정함(임의로 값을 넣어 봄)
w = tf.Variable(2.9)
b = tf.Variable(0.5)

# print(cost)

# learning_rate initialize
learning_rate = 0.01

# gradient descent
for i in range(100+1):
    with tf.GradientTape() as tape:
        # H(x) = Wx + b
        hypothesis = w * x_data + b
        # 랭크를 줄이고 평균을 구하는 함수 reduce_mean
        # 제곱을 해주는 함수 square
        cost = tf.reduce_mean(tf.square(hypothesis - y_data))

    w_grad, b_grad = tape.gradient(cost, [w, b])

    w.assign_sub(learning_rate * w_grad)
    b.assign_sub(learning_rate * b_grad)
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, w.numpy(), b.numpy(), cost))


# print(w * 5 + b)
# print(w * 2.5 + b)

plt.plot(x_data, y_data, 'o')
plt.plot(x_data, hypothesis.numpy(), 'r-')
plt.ylim(0, 8)