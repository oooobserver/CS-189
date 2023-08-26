import sys
sys.path.append('D:\Anaconda\Lib\site-packages')
import numpy as np
import matplotlib.pyplot as plt


data1 = np.load('dataset_1.npy')
data2 = np.load('dataset_2.npy')
x = data1[:, 0:1]
y = data1[:, 1:2]
x1 = data2[:, 0:1]
y1 = data2[:, 1:2]

def draw():
    plt.scatter(x, y, marker='o', color='red')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('XY Plot')
    plt.show()
    plt.scatter(x1, y1, marker='o', color='red')
    plt.show()



def correlation(x,y):
    u = np.sum(x)/len(x)
    v = np.sum(y)/len(y)
    cov_sum = 0
    dev_x = 0
    dev_y = 0
    for i in range(len(x)):
        cov_sum += (x[i]-u)*(y[i]-v)
        dev_x += (x[i]-u)**2
        dev_y += (y[i]-v)**2       
    cov =  cov_sum/len(x)
    dev = pow((dev_x/len(x)*dev_y/len(y)),1/2)
    return cov/dev




class data1:
    def __init__(self):
        column_temp = np.ones((len(x), 1))
        self.x_fi_1 = np.c_[x, column_temp]
        self.x_fi_2 = np.c_[x**2,x, column_temp]

        self.w = self.get_w(x)
        self.w1 = self.get_w(self.x_fi_1)
        self.w2 = self.get_w(self.x_fi_2)


    def get_w(self,matrix):
         return np.linalg.inv(matrix.T @ matrix) @ matrix.T @ y

    
    def check(self):
        column_temp = np.ones((len(x), 1))
        x_p = np.linspace(0, 10, num=100).reshape(-1, 1)
        y_p = x_p @ self.w

        x_p1 = np.c_[x_p, column_temp]
        y_p1 = x_p1 @ self.w1

        x_p2 = np.c_[x_p**2,x_p, column_temp]
        y_p2 = x_p2 @ self.w2


        fig, ax = plt.subplots()
        ax.plot(x_p, y_p, color='blue')
        ax.plot(x_p,y_p1, color='green')
        ax.plot(x_p,y_p2, color='yellow')

        ax.scatter(x, y, color='red')

        plt.show()

        m1 = x.T * self.w - y.T
        m2 = x * self.w - y
        print((m1 @ m2)/len(x))

        m1 = (self.x_fi_1 @ self.w1 - y).T
        m2 = self.x_fi_1 @ self.w1 - y
        print((m1 @ m2)/len(x))

        m1 = (self.x_fi_2 @ self.w2 - y).T
        m2 = self.x_fi_2 @ self.w2 - y
        print((m1 @ m2)/len(x))






class data2:
    def __init__(self):
        column_temp = np.ones((len(x1), 1))
        self.x_fi_1 = np.c_[x1, column_temp]
        self.x_fi_2 = np.c_[x1**2,x1, column_temp]

        self.w = self.get_w(x1)
        self.w1 = self.get_w(self.x_fi_1)
        self.w2 = self.get_w(self.x_fi_2)


    def get_w(self,matrix):
         return np.linalg.inv(matrix.T @ matrix) @ matrix.T @ y1


    def check(self):
        column_temp = np.ones((len(x1), 1))
        x_p = np.linspace(0, 10, num=100).reshape(-1, 1)
        y_p = x_p @ self.w

        x_p1 = np.c_[x_p, column_temp]
        y_p1 = x_p1 @ self.w1

        x_p2 = np.c_[x_p**2,x_p, column_temp]
        y_p2 = x_p2 @ self.w2

        fig, ax = plt.subplots()
        ax.plot(x_p, y_p, color='blue')
        ax.plot(x_p,y_p1, color='green')
        ax.plot(x_p,y_p2, color='yellow')

        ax.scatter(x1, y1, color='red')

        plt.show()

        m1 = x1.T * self.w - y1.T
        m2 = x1 * self.w - y1
        print((m1 @ m2)/len(x1))

        m1 = (self.x_fi_1 @ self.w1 - y1).T
        m2 = self.x_fi_1 @ self.w1 - y1
        print((m1 @ m2)/len(x1))

        m1 = (self.x_fi_2 @ self.w2 - y1).T
        m2 = self.x_fi_2 @ self.w2 - y1
        print((m1 @ m2)/len(x1))







def solution1():
    d = data1()
    d.check()
    d1 = data2()
    d1.check()





getW = lambda mx,my:np.linalg.inv(mx.T @ mx) @ mx.T @ my


sum = [0] * 5



def train(mx,my,vx=0,vy=0):
    lambda matrix: np.linalg.inv(matrix.T @ matrix) @ matrix.T @ y1
    size = len(mx)
    column_temp = np.ones((size, 1))
    xs = [0] * 5

    xs[0] = np.c_[mx, column_temp]
    xs[1] = np.c_[mx**2,xs[0]]
    xs[2] = np.c_[pow(mx,3) ,xs[1]]
    xs[3] = np.c_[pow(mx,4),xs[2]]
    xs[4] = np.c_[pow(mx,5),xs[3]]

    w = [0] * 5
    for i in range(5):
        w[i] = getW(xs[i],my)

    for i in range(5):
        m1 = (xs[i] @ w[i] - my).T
        m2 = xs[i] @ w[i] - my
        sum[i] += (m1 @ m2)/size





split_x= np.split(x1, 4)
split_y= np.split(y1, 4)



for i in range(4):
    x_val,y_val = split_x[i],split_y[i]

    X_train = split_x[:]
    del X_train[i]
    X_train = np.concatenate(X_train, axis=0)

    y_train = split_y[:]
    del y_train[i]
    y_train = np.concatenate(y_train, axis=0)

    train(X_train,y_train,x_val,y_val)



for i in sum:
    print(i/20)


















