# Category: Classifiers
In this folder I've build some simple classifiers, right now it is some simple experiments.

## CNNCIFARClissifier
This classifier model is a better version of the SimpleCIFARClassifier, where there have been introduced multiple convolutional layers to abstract more features from the image for the head of the model to learn from. This take more time to train due to the 3 block of Conv2D, BatchNorm2D, Conv2D, BatchNorm2D, MaxPool2D and Dropout. For more information on these types of layers I would refer you to keras documentation.

## CNNDigitClassifier
Here I build a simple CNN. Since mnist digits isn't a complicated image to process features throughout the network, there is no need for larger layers of convolutional layer. <br>
Using the package keras to help build, compile and fit the model. First it is needed to preprocess the data such that it will fit into the model build. Next the model is build, very simple with the base consisting of 2 Conv2D, 11 MaxPool2D and the head of 1 hidden layer and an output layer. This model perform quite well, given the input images is not more complex.

## SimpleCIFARClassifier
This is a classifier inspired from the networkClassifier, where the idea is the same, but without using any convolutional layer. As seen on these kind of images from CIFAR, which have much more complex features to abstract, it performs medium.

## networkClassifier
This classifier is just intro using only a few fully connected layers, i.e there is no convolutional layer. This classifier performs well on the mnist digits although the network does not abstract as many features to learn from. This python file outputs an image of the weights, just to give a visual representation on what your computer sees.