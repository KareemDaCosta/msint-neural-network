import random
from tqdm import tqdm
import numpy as np


class Network(object):
    def __init__(self, sizes):
        """
        Args:
            sizes (List[int]): Contains the size of each layer in the network.
        """
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]

    """
    4.1 Feed forward the input x through the network.
    """

    def feedforward(self, x):
        """
        Args:
            x (npt.array): Input to the network.
        Returns:
            List[npt.array]: List of weighted input values to each node
            List[npt.array]: List of activation output values of each node
        """
        inputs = []
        activationOutputs = [x]
        for layer in range(self.num_layers - 1):
            x = self.weights[layer] @ x + self.biases[layer]
            inputs.append(x)
            x = sigmoid(x)
            activationOutputs.append(x)
        return inputs, activationOutputs

    """
    4.2 Backpropagation to compute gradients.
    """

    def backprop(self, x, y, zs, activations):
        """
        Args:
            x (npt.array): Input vector.
            y (float): Target value.
            zs (List[npt.array]): List of weighted input values to each node.
            activations (List[npt.array]): List of activation output values of each node.
        Returns:
            List[npt.array]: List of gradients of bias parameters.
            List[npt.array]: List of gradients of weight parameters.
        """
        bias_gradients = [0 for i in range(self.num_layers - 1)]
        weight_gradients = [0 for i in range(self.num_layers - 1)]
        loss_derivative = self.loss_derivative(activations[-1], y)
        delta = loss_derivative * sigmoid_prime(zs[-1])  # delta l-1
        bias_gradients[-1] = delta
        weight_gradients[-1] = delta @ activations[-2].T

        for layer in range(self.num_layers - 3, -1, -1):
            delta = self.weights[layer+1].T @ delta * sigmoid_prime(zs[layer])
            bias_gradients[layer] = delta
            weight_gradients[layer] = np.outer(delta, activations[layer].T)
        return bias_gradients, weight_gradients

    """
    4.3 Update the network's biases and weights after processing a single mini-batch.
    """

    def update_mini_batch(self, mini_batch, alpha):
        """
        Args:
            mini_batch (List[Tuple]): List of (input vector, output value) pairs.
            alpha: Learning rate.
        Returns:
            float: Average loss on the mini-batch.
        """
        bias_sum = [np.zeros((y, 1)) for y in self.sizes[1:]]
        weight_sum = [np.zeros((y, x))
                      for x, y in zip(self.sizes[:-1], self.sizes[1:])]
        total_loss = 0
        for i in range(len(mini_batch)):
            zs, activations = self.feedforward(mini_batch[i][0])
            bias_gradients, weight_gradients = self.backprop(
                mini_batch[i][0], mini_batch[i][1], zs, activations)
            total_loss += self.loss_function(mini_batch[i][1], activations[-1])
            for j in range(len(bias_gradients)):
                bias_sum[j] += bias_gradients[j]
                weight_sum[j] += weight_gradients[j]
        for i in range(self.num_layers - 1):
            self.biases[i] -= alpha / len(mini_batch) * bias_sum[i]
            self.weights[i] -= alpha / len(mini_batch) * weight_sum[i]
        return total_loss / len(mini_batch)
    """
    Train the neural network using mini-batch stochastic gradient descent.
    """

    def SGD(self, data, epochs, alpha, decay, batch_size=32, test=None):
        n = len(data)
        losses = []
        for j in range(epochs):
            print(f"training epoch {j+1}/{epochs}")
            random.shuffle(data)
            for k in tqdm(range(n // batch_size)):
                mini_batch = data[k * batch_size: (k + 1) * batch_size]
                loss = self.update_mini_batch(mini_batch, alpha)
                losses.append(loss)
            alpha *= decay
            if test:
                print(f"Epoch {j+1}: eval accuracy: {self.evaluate(test)}")
            else:
                print(f"Epoch {j+1} complete")
        return losses

    """
    Returns classification accuracy of network on test_data.
    """

    def evaluate(self, test_data):
        test_results = [
            (np.argmax(self.feedforward(x)[1][-1]), y) for (x, y) in test_data
        ]
        return sum(int(x == y) for (x, y) in test_results) / len(test_data)

    def loss_function(self, y, y_prime):
        return 0.5 * np.sum((y - y_prime) ** 2)

    """
    Returns the gradient of the squared error loss function.
    """

    def loss_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
