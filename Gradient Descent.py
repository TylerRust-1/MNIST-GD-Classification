from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

# Hyperparameters for training/testing
n_train = 10000  # Number of training data to train classifier Max of 60000
n_test = 1000  # Number of test data to test classifier Max of 10000
r = 2 # Regularization parameter
n = n_train + n_test

# Load data from mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x = np.append(x_train, x_test)
# y = np.append(y_train, y_test)

x_train = x_train[:n_train, :]
x_test = x_test[:n_test, :]
y_train = y_train[:n_train]
y_test = y_test[:n_test]  # Working

print("Initial Data Format")
print('x_train: ' + str(x_train.shape))
print('y_train: ' + str(y_train.shape))
print('x_test:  ' + str(x_test.shape))
print('y_test:  ' + str(y_test.shape))

x_train = x_train.reshape(n_train, 784)
x_test = x_test.reshape(n_test, 784)

x_train = np.round(x_train / 255, 5)

x_train = x_train.T
x_test = x_test.T

print("\nReformatted data - Rows are features, columns are data points.")
print('x_train: ' + str(x_train.shape))
print('y_train: ' + str(y_train.shape))
print('x_test:  ' + str(x_test.shape))
print('y_test:  ' + str(y_test.shape))
print()

# This shenanigans is O(N), probably a better way to do this.
# Gives me 10 different vectors to perform binary classification for each number.
y_train0 = np.zeros(n_train)
y_train1 = np.zeros(n_train)
y_train2 = np.zeros(n_train)
y_train3 = np.zeros(n_train)
y_train4 = np.zeros(n_train)
y_train5 = np.zeros(n_train)
y_train6 = np.zeros(n_train)
y_train7 = np.zeros(n_train)
y_train8 = np.zeros(n_train)
y_train9 = np.zeros(n_train)
for i, value in enumerate(y_train):
    if value == 0:
        y_train0[i] = 1
    if value == 1:
        y_train1[i] = 1
    if value == 2:
        y_train2[i] = 1
    if value == 3:
        y_train3[i] = 1
    if value == 4:
        y_train4[i] = 1
    if value == 5:
        y_train5[i] = 1
    if value == 6:
        y_train6[i] = 1
    if value == 7:
        y_train7[i] = 1
    if value == 8:
        y_train8[i] = 1
    if value == 9:
        y_train9[i] = 1


# Functions for calculating the gradient
def sigmoid(theta, xi):
    x = -np.inner(theta, xi)
    if x > 300:  # To prevent overflow error (chose 300 because accuracy stayed same
        return 0
    return 1 / (1 + np.exp(x))


def grad(theta, x_train, y_train):
    global r #regularization parameter
    gradient = 2 * r * theta
    hessian = 2 * r * np.eye(784)
    for i in range(n_train):
        current_data = x_train[:, i]
        gradient += (current_data * (sigmoid(theta, current_data) - y_train[i]))
        hessian += np.outer(current_data, current_data) * (sigmoid(theta, current_data)) * (
                1 - sigmoid(theta, current_data))
    return gradient, hessian


theta0 = np.zeros((784,))
theta1 = np.zeros((784,))
theta2 = np.zeros((784,))
theta3 = np.zeros((784,))
theta4 = np.zeros((784,))
theta5 = np.zeros((784,))
theta6 = np.zeros((784,))
theta7 = np.zeros((784,))
theta8 = np.zeros((784,))
theta9 = np.zeros((784,))
for i in range(2):
    print("iteration", i + 1)
    g0, h0 = grad(theta0, x_train, y_train0)
    g1, h1 = grad(theta1, x_train, y_train1)
    g2, h2 = grad(theta2, x_train, y_train2)
    g3, h3 = grad(theta3, x_train, y_train3)
    g4, h4 = grad(theta4, x_train, y_train4)
    g5, h5 = grad(theta5, x_train, y_train5)
    g6, h6 = grad(theta6, x_train, y_train6)
    g7, h7 = grad(theta7, x_train, y_train7)
    g8, h8 = grad(theta8, x_train, y_train8)
    g9, h9 = grad(theta9, x_train, y_train9)

    theta0 -= np.linalg.inv(h0).dot(g0)
    theta1 -= np.linalg.inv(h1).dot(g1)
    theta2 -= np.linalg.inv(h2).dot(g2)
    theta3 -= np.linalg.inv(h3).dot(g3)
    theta4 -= np.linalg.inv(h4).dot(g4)
    theta5 -= np.linalg.inv(h5).dot(g5)
    theta6 -= np.linalg.inv(h6).dot(g6)
    theta7 -= np.linalg.inv(h7).dot(g7)
    theta8 -= np.linalg.inv(h8).dot(g8)
    theta9 -= np.linalg.inv(h9).dot(g9)

y_pred = []
for i in range(n_train):
    current_data = x_train[:, i]
    temp = (sigmoid(theta0, current_data))
    largest = 0
    '''
    print(sigmoid(theta0, current_data))
    print(sigmoid(theta1, current_data))
    print(sigmoid(theta2, current_data))
    print(sigmoid(theta3, current_data))
    print(sigmoid(theta4, current_data))
    print(sigmoid(theta5, current_data))
    print(sigmoid(theta6, current_data))
    print(sigmoid(theta7, current_data))
    print(sigmoid(theta8, current_data))
    print(sigmoid(theta9, current_data))
    '''
    if sigmoid(theta1, current_data) > temp:
        temp = sigmoid(theta1, current_data)
        largest = 1
    if sigmoid(theta2, current_data) > temp:
        temp = sigmoid(theta2, current_data)
        largest = 2
    if sigmoid(theta3, current_data) > temp:
        temp = sigmoid(theta3, current_data)
        largest = 3
    if sigmoid(theta4, current_data) > temp:
        temp = sigmoid(theta4, current_data)
        largest = 4
    if sigmoid(theta5, current_data) > temp:
        temp = sigmoid(theta5, current_data)
        largest = 5
    if sigmoid(theta6, current_data) > temp:
        temp = sigmoid(theta6, current_data)
        largest = 6
    if sigmoid(theta7, current_data) > temp:
        temp = sigmoid(theta7, current_data)
        largest = 7
    if sigmoid(theta8, current_data) > temp:
        temp = sigmoid(theta8, current_data)
        largest = 8
    if sigmoid(theta9, current_data) > temp:
        temp = sigmoid(theta9, current_data)
        largest = 9
    y_pred.append(largest)
y_pred = np.array(y_pred)

print("Training Accuracy: ", round((sum(y_pred == y_train) / n_train) * 100, 2), "%")

y_pred = []
for i in range(n_test):
    current_data = x_test[:, i]
    temp = (sigmoid(theta0, current_data))
    largest = 0
    if sigmoid(theta1, current_data) > temp:
        temp = sigmoid(theta1, current_data)
        largest = 1
    if sigmoid(theta2, current_data) > temp:
        temp = sigmoid(theta2, current_data)
        largest = 2
    if sigmoid(theta3, current_data) > temp:
        temp = sigmoid(theta3, current_data)
        largest = 3
    if sigmoid(theta4, current_data) > temp:
        temp = sigmoid(theta4, current_data)
        largest = 4
    if sigmoid(theta5, current_data) > temp:
        temp = sigmoid(theta5, current_data)
        largest = 5
    if sigmoid(theta6, current_data) > temp:
        temp = sigmoid(theta6, current_data)
        largest = 6
    if sigmoid(theta7, current_data) > temp:
        temp = sigmoid(theta7, current_data)
        largest = 7
    if sigmoid(theta8, current_data) > temp:
        temp = sigmoid(theta8, current_data)
        largest = 8
    if sigmoid(theta9, current_data) > temp:
        temp = sigmoid(theta9, current_data)
        largest = 9
    y_pred.append(largest)
# print("Predicted values\n", y_pred)
# print("Actual values\n", y_test)
print("Number of training data:", n_train)
print("Number of test data:", n_train)
print("Regularization parameter r:", r)
print("Prediction and test accuracy: ", round((sum(y_pred == y_test) / n_test) * 100, 2), "%")
print("Test Error: ", round((sum(y_pred != y_test) / n_test) * 100, 2), "%")

temp = []
for i in range(n_test):
    if y_pred[i] != y_test[i]:
        temp.append(i)

for i in temp:
    image = x_test[:, i].reshape(28, 28)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.title("Prediction: " + str(y_pred[i]) + " Actual: " + str(y_test[i]))
    plt.show()
