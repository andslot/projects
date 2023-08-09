import matplotlib.pyplot as plt
import numpy as np

weight_value = 100

bias_value_1 = 30

bias_value_2 = -50

plt.axis([-10, 10, -1, 10])

print("The step function starts at {0} and ends at {1}".format(-bias_value_1 / weight_value, -bias_value_2 / weight_value))

inputs = np.arange(-10, 10, 0.01)
outputs = list()

for x in inputs:
    y1 = 1.0 / (1.0 + np.exp(-weight_value * x - bias_value_1))
    y2 = 1.0 / (1.0 + np.exp(-weight_value * x - bias_value_2))

    w = 2

    y = y1 * w - y2 * w

    outputs.append(y)

plt.plot(inputs, outputs, lw=2, color ='black')
plt.show()