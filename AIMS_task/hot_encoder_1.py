import numpy as np

class OneHotEncoder:
  def __init__(self, categories):
    self.categories = categories
    self.num_categories = len(categories)
    self.encoding_matrix = np.zeros((self.num_categories, self.num_categories))
    for i, category in enumerate(categories):
      self.encoding_matrix[i][i] = 1

  def encode(self, data):
    #Encode a list of categorical values using the one-hot encoding matrix
    encoded_data = np.zeros((len(data), self.num_categories))
    for i, value in enumerate(data):
      if value in self.categories:
        encoded_data[i] = self.encoding_matrix[self.categories.index(value)]
      else:
        raise ValueError(f"Value '{value}' not found in categories.")
    return encoded_data

# Example usage
categories = [] #import data categories here
encoder = OneHotEncoder(categories)
data = [] #import data here
encoded_data = encoder.encode(data)

print(encoded_data)