from distutils.log import error
import pandas as pd
import torch
import numpy as np
import tensorflow as tf

# Download flower dataset
dataset = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species'])

# Set index of species/catagories
dataset['species'] = pd.Categorical(dataset['species']).codes

#shuffle rows in dataset
dataset = dataset.sample(frac = 1, random_state = 1234)

# Take first 120 as training data
train_input = dataset.values[:120, :4]
train_target = dataset.values[:120, 4]

# Take last 30 as testing data
test_input = dataset.values[120:, :4]
test_target = dataset.values[120:, 4]



torch.manual_seed(1234)

hidden_units = 5

# Creating a neural network with 1 input layer of 4 units, 1 hidden layer with 5 units, 1 output layer of 3 units, hidden layer using ReLU activation function
net = torch.nn.Sequential(
    torch.nn.Linear(4, hidden_units), 
    torch.nn.ReLU(), 
    torch.nn.Linear(hidden_units, 3))


# Choose optimizer and loss function
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)



# Train
epochs = 50

for epoch in range(epochs):
    inputs = torch.autograd.Variable(torch.Tensor(train_input).float())
    targets = torch.autograd.Variable(torch.Tensor(train_target).long())

    optimizer.zero_grad()
    out = net(inputs)

    loss = criterion(out, targets)
    loss.backward()
    optimizer.step()

    if epoch == 0 or (epoch + 1) % 10 == 0:
        print('Epoch %d Loss: %.4f' % (epoch + 1, loss.item()))



# (TEST) Accuracy of model
inputs = torch.autograd.Variable(torch.Tensor(test_input).float())
targets = torch.autograd.Variable(torch.Tensor(test_target).long())

optimizer.zero_grad()
out = net(inputs)
_, predicted = torch.max(out.data, 1)

error_count = test_target.size - np.count_nonzero((targets == predicted).numpy())
print('Errors: %d; Accuracy: %d%%' % (error_count, (100 * torch.sum(targets == predicted) / test_target.size)))