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
# %%
