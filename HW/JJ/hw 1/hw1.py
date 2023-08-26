import sys
sys.path.append('D:\Anaconda\Lib\site-packages')
import numpy as np
import matplotlib.pyplot as plt
import pickle

fileNames = ["x_train.p","y_train.p","x_test.p","y_test.p"]
dataSets = [0] * 4
for i in range(4):
    with open(fileNames[i], 'rb') as file:
        dataSets[i] = pickle.load(file)





def vualize():
    images_to_show = [dataSets[0][0], dataSets[0][10], dataSets[0][20]]
    fig,axes = plt.subplots(1, 3, figsize=(6, 6))
    for i, image in enumerate(images_to_show):
        corrected_image = np.copy(image)
        corrected_image[:, :, [0, 2]] = corrected_image[:, :, [2, 0]]

        axes[i].imshow(corrected_image)  
        axes[i].set_title(f"{i}th ")
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()


def compose(X):
    num_images = X.shape[0]
    flattened_images = X.reshape(num_images, -1)
    return flattened_images 


def standard_compose(X):
    flattened_images = compose(X)
    flattened_images = flattened_images/255.0*2-1
    return flattened_images 


def OLS(X,Y):
    try:
        return np.linalg.inv(X.T @ X) @ X.T @ Y
    except np.linalg.LinAlgError as e:
        print("Error:", e)


def ASED(x,w,u):
    row = x.shape[0]
    sum = 0
    for i in range(row):
        sum += np.linalg.norm(x[i] @ w - u[i])
    return sum / row


def Ridge(X,Y):
    a = [0.1, 1.0, 10, 100, 1000]
    row,column = X.shape
    I = np.eye(column,column)
    for i in range(len(a)):
        w = np.linalg.inv(X.T @ X + a[i]*I) @ X.T @ Y
        print(f'{a[i]}:')
        print(ASED(X,w,Y))


class solution:
    def a():
        vualize()

    def b():
        X = compose(dataSets[0])
        OLS(X,dataSets[1])

    def c():
        X = compose(dataSets[0])
        Ridge(X,dataSets[1])

    def d():
        X = standard_compose(dataSets[0])
        OLS(X,dataSets[1])
        Ridge(X,dataSets[1])

    def e():
        X = compose(dataSets[2])
        X_standard = standard_compose(dataSets[2])
        OLS(X,dataSets[3])
        Ridge(X,dataSets[3])
        OLS(X_standard,dataSets[3])
        Ridge(X_standard,dataSets[3])

    def f():
        X = compose(dataSets[2])
        X_standard = standard_compose(dataSets[2])
        row,column = X.shape
        I = np.eye(column,column)
        m  = X.T @ X + 100*I
        m1 = X_standard.T @ X_standard + 100*I

        singular_values1 = np.linalg.svd(m, compute_uv=False)
        singular_values2 = np.linalg.svd(m1, compute_uv=False)
        print(f'without standardiz k is {singular_values1[0]/singular_values1[-1]}')
        print(f'standardiz k is {singular_values2[0]/singular_values2[-1]}')

    def s():
        solution.a()
        solution.b()
        solution.c()
        solution.d()
        solution.e()
        solution.f()




solution.s()

    


