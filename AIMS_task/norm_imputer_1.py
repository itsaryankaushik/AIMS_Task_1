import numpy as np

class CustomSimpleImputer:
  ##using callbacks for stategy choice
  def __init__(self, missing_values=np.nan, strategy_callback=None, fill_value=None,
               copy=True, add_indicator=False, keep_empty_features=False):
    self.missing_values = missing_values
    self.strategy_callback = strategy_callback
    self.fill_value = fill_value
    self.copy = copy
    self.add_indicator = add_indicator
    self.keep_empty_features = keep_empty_features
    self.statistics_ = None

  def fit(self, X, y=None):
    if self.copy:
      X = X.copy()
    self.statistics_ = {}
    for col in range(X.shape[1]):
      missing_mask = X[:, col] == self.missing_values
      if missing_mask.any():
        if self.strategy_callback:
          self.statistics_[col] = self.strategy_callback(X[~missing_mask, col])
        else:
          raise ValueError("No strategy callback provided for missing values.")
    return self

  def transform(self, X):
    #Transform the data by imputing missing values
    if self.copy:
      X = X.copy()
    for col, stats in self.statistics_.items():
      missing_mask = X[:, col] == self.missing_values
      if missing_mask.any():
        replacement = np.full_like(X[:, col], self.fill_value)
        replacement[missing_mask] = stats
      X[:, col] = np.where(missing_mask, replacement, X[:, col])
    if self.add_indicator:
      missing_indicators = np.zeros((X.shape[0], X.shape[1]))
      for col in range(X.shape[1]):
        missing_indicators[:, col] = (X[:, col] == self.missing_values).astype(int)
      X = np.c_[X, missing_indicators]
    if not self.keep_empty_features:
      return X[:, ~np.all(X == self.missing_values, axis=0)]
    return X

# Example usage
def custom_mean(data):
  return np.mean(data)

imputer = CustomSimpleImputer(strategy_callback=custom_mean)
data = np.array([[1, 2, 3], [np.nan, 4, 5], [6, 7, np.nan]])
imputer.fit(data)
transformed_data = imputer.transform(data)

print(transformed_data)