from keras.datasets import mnist
from keras.models import Sequential
import keras.layers as nn
from keras.utils import to_categorical
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(60000, 28, 28, 1)
X_test = X_test.reshape(10000, 28, 28, 1)

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# build model
model = Sequential([
    nn.Conv2D(filters=32, kernel_size=(3, 3), input_shape=(28, 28, 1), activation='relu'),
    nn.Conv2D(filters=32, kernel_size=(3, 3), activation='relu'),
    nn.MaxPooling2D(pool_size=(2, 2)),
    nn.Flatten(),
    nn.Dense(64, activation='relu'),
    nn.Dense(10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(X_train, y_train, batch_size=100, epochs=5, validation_split=0.1, verbose=1)

score = model.evaluate(X_test, y_test, verbose=1)
print(f'Test accuracy: {score[1]}')
