# Generating Images with AI

## GeneratingMNISTDigitsWithVAE
Here I use a Variational AutoEncoder. The idea behind these autoencoders is that we have a feed-forward neural network, that tried to reproduce its input. Thus the target is equal to the input (when training the model). An autoencoder is an unsupervised algorithm, since the "labels" are just the input data. <br>
Usually autoencoders consist of input - hidden layer - output. The hidden layer is a bottleneck, hence there is fewer neurons than the size of the input. <br>
Training such a model we would use a reconstruction error L = (x, x'). Which measures the distance between the original input and its reconstruction. It can be minimized in the usaul way with gradient descent and backpropagation. The need of the variational comes in hand when we do not want the encoder to map each input sample to the latent space and each attribute of the latent representation has a discrete value. <br>
<br>
What we want it to generate new images that are different from the original. <br>
The basis is that we do not want the bottleneck layer to directly output the latent state variables, but instead output two vectors, which describe the mean and variance of the distribution of each latent variable. This has its problems since we cannot backpropegate over random processes. To fix this we use reparameterization trick, where we sample a random vector, epsilon, with the same dimensions as our sample stat, from a Gaussian distribution. Then we shift it by the latent distribution's mean and scale it by the latent distribution's  variance. <br>
To calculate the loss with reconstruction loss from cross-entropy and Kullback-Leiber divergence.