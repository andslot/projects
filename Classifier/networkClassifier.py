from keras.models import Sequential
from keras.layers import Dense, Activation, CategoryEncoding
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm


def one_in_k_encoding(vec, k):
    """ One-in-k encoding of vector to k classes

    Args:
       vec: numpy array - data to encode
       k: int - number of classes to encode to (0,...,k-1)
    """
    n = vec.shape[0]
    enc = np.zeros((n, k))
    enc[np.arange(n), vec] = 1
    return enc



(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

# Reshape input data to vectors of 28x28=784 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

# One hot encode labels
classes = 10
Y_train = one_in_k_encoding(Y_train, classes)
Y_test = one_in_k_encoding(Y_test, classes)

# Hyper parameters, defined batch size, number of hidden neurons, epochs etc...
input_size = 784
batch_size = 100
hidden_neurons = 100
epochs = 100

# Define our model with one hidden layer, being fully-connected layer, sigmoid activation, and then softmax output
model = Sequential([
    Dense(hidden_neurons, input_dim=input_size),
    Activation('sigmoid'),
    Dense(classes),
    Activation('softmax')
])

# Compile model with loss of crossentropy since we use one-hot encoded and softmax, use stocastic gradient descent as optimizer
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='sgd')

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1)

score = model.evaluate(X_test, Y_test, verbose=1)
print('Test accuracy:', score[1])


# Visualization of what happens with the weights
weights = model.layers[0].get_weights()

fig = plt.figure()

w = weights[0].T
for neuron in range(hidden_neurons):
    ax = fig.add_subplot(10, 10, neuron + 1)
    ax.axis("off")
    ax.imshow(np.reshape(w[neuron], (28, 28)), cmap=cm.get_cmap('Greys'))

plt.savefig("neuron_images.png", dpi=300)
plt.show()