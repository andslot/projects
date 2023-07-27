from keras.datasets import cifar10
from keras.layers import Dense, Activation
from keras.models import Sequential
from keras.utils import to_categorical
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
import random

(X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

X_train = X_train.reshape(50000, 3072)
X_test = X_test.reshape(10000, 3072)

classes = 10
Y_train = to_categorical(Y_train, classes)
Y_test = to_categorical(Y_test, classes)

input_size = 3072
batch_size = 100
epochs = 100

# Build model 3 hidden layers for a more complex task and rather large hidden layers
model = Sequential([
    Dense(1024, input_dim=input_size, activation='relu'),
    Dense(512, activation='relu'),
    Dense(512, activation='sigmoid'),
    Dense(classes, activation='softmax'),
])

model.compile(loss='categorical_crossentropy',
              metrics=['accuracy'], optimizer='sgd')
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), verbose=1)


# Visualize
fig = plt.figure()
outer_grid = gridspec.GridSpec(10, 10, wspace=0.0, hspace=0.0)

weights = model.layers[0].get_weights()

w = weights[0].T

for i, neuron in enumerate(random.sample(range(0, 1023), 100)):
    ax = plt.Subplot(fig, outer_grid[i])
    ax.imshow(np.mean(np.reshape(w[i], (32, 32, 3)), axis=2), cmap=cm.get_cmap('Greys'))
    ax.set_xticks([])
    ax.set_yticks([])
    fig.add_subplot(ax)

plt.show()
