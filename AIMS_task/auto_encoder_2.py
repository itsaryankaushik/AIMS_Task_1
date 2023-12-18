import numpy as np

class Autoencoder:
  def __init__(self, input_dim, hidden_dim):
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.W1 = np.random.randn(self.input_dim, self.hidden_dim)
    self.b1 = np.zeros(self.hidden_dim)
    self.W2 = np.random.randn(self.hidden_dim, self.input_dim)
    self.b2 = np.zeros(self.input_dim)

  def local_func(self, x):
    script = np.sin(2 * np.pi * x[:, 1])
    return x + script[:, np.newaxis]

  def encode(self, x):
    x = self.local_func(x)
    h = np.tanh(np.dot(x, self.W1) + self.b1)
    return h

  def decode(self, h):
    x_recon = np.tanh(np.dot(h, self.W2) + self.b2)
    return x_recon

  def train(self, X, learning_rate=0.01, epochs=100):
    for epoch in range(epochs):
      for i in range(len(X)):
        x = X[i]
        h = self.encode(x)
        x_recon = self.decode(h)
        # Calculate reconstruction error
        error = x_recon - x
        # Update weights with backpropagation
        delta_output = error * (1 - x_recon**2)
        delta_hidden = delta_output.dot(self.W2.T) * (1 - h**2)
        self.W2 -= learning_rate * np.outer(h, delta_output)
        self.b2 -= learning_rate * delta_output
        self.W1 -= learning_rate * np.outer(x, delta_hidden)
        self.b1 -= learning_rate * delta_hidden

# Example usage
input_dim = 2
hidden_dim = 5
data = np.random.randn(100, input_dim)
autoencoder = Autoencoder(input_dim, hidden_dim)
autoencoder.train(data)
encoded_data = autoencoder.encode(data)
reconstructed_data = autoencoder.decode(encoded_data)

print("Original data:", data)
print("Reconstructed data:", reconstructed_data)


# encode: This method takes an input data point and encodes it 
# through the hidden layer using the trained weights and biases.

## here it adds a simple sine wave to the first dimension based on the second dimension
# Before encoding, it applies the local script function to manipulate
# the data based on your specific logic.
# decode: This method takes the encoded representation and decodes
#  it back to the original data space.