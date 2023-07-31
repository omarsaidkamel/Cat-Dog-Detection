# import library
# use the array and math
import numpy as np
# show dataset
import matplotlib.pyplot as plt
# choose random
import random
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPool2D, Dense, Flatten



# get the train data
x_train = np.loadtxt("C:\\Users\\oahly\\Downloads\\input.csv", delimiter=',')
y_train = np.loadtxt('C:\\Users\\oahly\\Downloads\\labels.csv', delimiter=',')

# get the test data
x_test = np.loadtxt('C:\\Users\\oahly\\Downloads\\input_test.csv', delimiter=',')
y_test = np.loadtxt('C:\\Users\\oahly\\Downloads\\labels_test.csv', delimiter=',')

# Reshape train data
x_train = x_train.reshape(len(x_train), 100, 100, 3)
y_train = y_train.reshape(len(y_train), 1)

# Reshape test data
x_test = x_test.reshape(len(x_test), 100, 100, 3)
y_test = y_test.reshape(len(y_test), 1)

# Rescale values
x_train = x_train / 255
x_test = x_test / 255

# sequence of model (layers of model)
model = Sequential([
    # 32 -> number of filter
    # (3,3) -> size of filter
    # activation='relu' -> type of activation function
    # input_shape=(100, 100, 3)-> expected input shap
    Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    # (2,2) -> filter size
    MaxPool2D((2, 2)),

    Conv2D(32, (3, 3), activation='relu'),
    MaxPool2D((2, 2)),

    Flatten(),  # convert matrix to vector to be NN
    # fully connected layer 64 -> neurons
    Dense(64, activation='relu'),
    # 1 -> output neurons
    # use sigmoid because it is used for binary classification
    Dense(1, activation='sigmoid'),
])

# define model in another way
# model = Sequential()
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3))),
# model.add(MaxPool2D((2, 2)),)
# model.add(Conv2D(32, (3, 3), activation='relu', ),)
# model.add(MaxPool2D((2, 2)),)
# model.add(Flatten(),)
# model.add(Dense(64, activation='relu'),)
# model.add(Dense(1, activation='Sigmoid'),)

# optimizer = tf.python.keras.optimizers.SGD(learning_rate=0.001)

# add cost - back propagation algorithm
model.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

# fit input data
model.fit(x_train, y_train, epochs=5, batch_size=64)

# test data set accuracy
model.evaluate(x_test, y_test)

# show random image
idx = random.randint(0, len(x_test))

y_pred = model.predict(x_test[idx,:].reshape(1, 100, 100, 3))

y_pred = y_pred > 0.5

if (y_pred == 0):
    pred = 'Dog'
else:
    pred = 'Cat'

print('The model predict -> ', pred)

plt.imshow(x_test[idx])
plt.show()