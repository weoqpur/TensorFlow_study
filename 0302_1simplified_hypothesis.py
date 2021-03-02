import numpy as np

X = np.array([1, 2, 3])
Y = np.array([1, 2, 3])

def cost_func(W, X, Y):
    c = 0 # 합을 구할 변수
    for i in range(len(X)): # 배열의 길이 만큼 반복문 실행
        c += (W * X[i] - Y[i]) ** 2 # (Wxi - yi)**2
    return c / len(X)
 # np.linspace(x, y, num=z): x부터 y까지 15개로 나누는 함수
for feed_W in np.linspace(-3, 5, num=15):
    curr_cost = cost_func(feed_W, X, Y)
    print("{:6.3f} | {:10.5f}".format(feed_W, curr_cost))
