import tensorflow as tf
from keras.datasets import mnist
from matplotlib import pyplot as plt
import numpy as np

# Hyperparameters for training/testing
n_train = 60000  # Number of training data to train classifier Max of 60000
n_test = 10000  # Number of test data to test classifier Max of 10000
n = n_train + n_test

# Load data from mnist dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

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

x_train = x_train / 255
x_test = x_test / 255

print("\nReformatted data - Rows are features, columns are data points.")
print('x_train: ' + str(x_train.shape))
print('y_train: ' + str(y_train.shape))
print('x_test:  ' + str(x_test.shape))
print('y_test:  ' + str(y_test.shape))
print()

model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(784,), activation='sigmoid')  # The input shape is 784.
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)

y_pred = model.predict(x_test)
final_pred = []
for i in y_pred:
    final_pred.append(np.argmax(i))

print("Number of training data:", n_train)
print("Number of test data:", n_test)
print("Prediction and test accuracy: ", round((sum(final_pred == y_test) / n_test) * 100, 2), "%")
print("Test Error: ", round((sum(final_pred != y_test) / n_test) * 100, 2), "%")

temp = []
for i in range(n_test):
    if final_pred[i] != y_test[i]:
        temp.append(i)

for i in temp:
    image = x_test[i, :].reshape(28, 28)
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.title("Prediction: " + str(final_pred[i]) + " Actual: " + str(y_test[i]))
    plt.show()
