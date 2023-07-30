import keras
from keras.datasets import cifar10
import keras.layers as layer
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

# define mini-batch size for covariance
batch_size = 50

# load cifar-10 and normalize data by dividing it to 255 (maximum pixel intensity)
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# apply data augmentation
data_generator = ImageDataGenerator(rotation_range=90,
                                    width_shift_range=0.1,
                                    height_shift_range=0.1,
                                    featurewise_center=True,
                                    featurewise_std_normalization=True,
                                    horizontal_flip=True)

data_generator.fit(x_train)

# Standardize the test set
for i in range(len(x_test)):
    x_test[i] = data_generator.standardize(x_test[i])

model = Sequential([
    layer.Conv2D(filters=32, kernel_size=(3, 3), padding='same', input_shape=(x_train.shape[1:])),
    layer.BatchNormalization(),
    layer.Activation('elu'),
    layer.Conv2D(filters=32, kernel_size=(3, 3), padding='same'),
    layer.BatchNormalization(),
    layer.Activation('elu'),
    layer.MaxPool2D(pool_size=(2, 2)),
    layer.Dropout(rate=0.2),

    layer.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
    layer.BatchNormalization(),
    layer.Activation('elu'),
    layer.Conv2D(filters=64, kernel_size=(3, 3), padding='same'),
    layer.BatchNormalization(),
    layer.Activation('elu'),
    layer.MaxPool2D(pool_size=(2, 2)),
    layer.Dropout(rate=0.2),

    layer.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
    layer.BatchNormalization(),
    layer.Activation('elu'),
    layer.Conv2D(filters=128, kernel_size=(3, 3), padding='same'),
    layer.BatchNormalization(),
    layer.Activation('elu'),
    layer.MaxPool2D(pool_size=(2, 2)),
    layer.Dropout(rate=0.2),

    layer.Flatten(),
    layer.Dense(units=10, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# fit the model on the generator which consists of original train set images but also augmented, validate on the test set to know the acutal accuracy
model.fit(data_generator.flow(x=x_train, y=y_train, batch_size=batch_size),
          steps_per_epoch=len(x_train) // batch_size,
          epochs=100,
          validation_data=(x_test, y_test),
          workers=4)
