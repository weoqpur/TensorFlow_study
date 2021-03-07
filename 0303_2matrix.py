import tensorflow as tf
import numpy as np

data = np.array([
    [ 73., 80., 75., 152.],
    [ 93., 88., 93., 185.],
    [ 89., 91., 90., 180.],
    [ 96., 98.,100., 196.],
    [ 73., 66., 70., 142.]
], dtype=np.float32)

X = data[:, :-1] # 마지막만 출력 X
Y = data[:, -1] # 마지막만 출력

# 랜덤으로 리스트 생성 3개의 리스트에 한 개의 숫자
w = tf.Variable(tf.random.normal([3, 1]))
# 랜덤한 한 개의 숫자
b = tf.Variable(tf.random.normal([1]))

learning_rate = 0.000001

def predict(X):
    return tf.matmul(X, w) + b

n_epochs = 2000
for i in range(n_epochs + 1):
    with tf.GradientTape() as tape:
        cost = tf.reduce_mean((tf.square(predict(X) - Y)))

    w_grad, b_grad = tape.gradient(cost, [w, b])

    # W = W - learning_rate * w_grad
    w.assign_sub(learning_rate * w_grad)
    b.assign_sub(learning_rate * b_grad)

    if i % 100 == 0:
        print("{:5} | {:10.4f}".format(i, cost.numpy()))