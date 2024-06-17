import numpy as np
from implemented_xgboost import XGBoost

# Sample data
X = np.array([[1, 2], [1, 3], [1, 4], [2, 2], [2, 3], [2, 4], [3, 2], [3, 3], [3, 4]])
y = np.array([1, 1, 1, -1, -1, -1, 1, 1, 1])

# print(y.dtype)

# Initialize and train the model
model = XGBoost(n_estimators=10, learning_rate=0.1, max_depth=3, min_samples_split=2)
model.fit(X, y)

# Make predictions
predictions = model.predict(X)
print("Predictions:", predictions)