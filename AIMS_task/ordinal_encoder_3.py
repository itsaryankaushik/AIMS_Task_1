import numpy as np
class OrdinalEncoder:
  def __init__(self, categories):
    self.categories = categories
    self.encoding_map = {category: i for i, category in enumerate(categories)}

  def transform(self, data):
    encoded_data = []
    for value in data:
      if value in self.categories:
        encoded_data.append(self.encoding_map[value])
      else:
        raise ValueError(f"Value '{value}' not found in categories.")
    return np.array(encoded_data)

# Same usage as before
categories = [] #pass data categories here 
#["low", "medium", "high"]
encoder = OrdinalEncoder(categories)
data = [] #pass data here
#["high", "low", "medium", "medium", "high"]
encoded_data = encoder.transform(data)

print(encoded_data)  # Output: [[2], [0], [1], [1], [2]]

# It allows for more control over the encoding process and handling of unknown categories