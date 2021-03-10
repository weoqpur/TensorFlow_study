#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

x_train = [[1., 2.], [2., 3.], [3., 1.], [4., 3.], [5., 3.], [6., 2.]]
y_train = [[0.], [0.], [0.], [1.], [1.], [1.]]

x_test = [[5., 2.]]
y_test = [[1.]]

# x_train 데이터를 반복문으로 x1에는 첫번째 값 x2에는 두번째 값 저장
x1 = [x[0] for x in x_train]
x2 = [x[1] for x in x_train]

# 1과 0을 색으로 분류
colors = [int(y[0] % 3) for y in y_train]
plt.scatter(x1,x2, c=colors , marker='^')
# 테스트 데이터는 빨간 색으로 표시
plt.scatter(x_test[0][0],x_test[0][1], c="red")

# 데이터 시각화
plt.xlabel("x1")
plt.xlabel("x2")
plt.show()

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(len(x_train))#.repeat()

W = tf.Variable(tf.zeros([2,1]), name = 'weight')
B = tf.Variable(tf.zeros([1]), name='bias')

def logistic_regression(features):
    hypothesis = tf.divide(1., 1. + tf.exp(tf.matmul(features, W) + B))
    return hypothesis

def loss_fn(hypothesis, labels):
    cost = -tf.reduce_mean(labels * tf.math.log(hypothesis) + (1 - labels) * tf.math.log(1 - hypothesis))
    return cost

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

def accuracy_fn(hypothesis, labels):
    predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
    accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, labels), dtype=tf.int32))
    return accuracy
# %%
