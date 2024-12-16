
import numpy as np
import matplotlib.pyplot as plt

# Generate a simple binary dataset
np.random.seed(42)
num_samples = 100
x1_class_0 = np.random.normal(2, 1, num_samples // 2)
x2_class_0 = np.random.normal(2, 1, num_samples // 2)
x1_class_1 = np.random.normal(6, 1, num_samples // 2)
x2_class_1 = np.random.normal(6, 1, num_samples // 2)

x = np.vstack((np.c_[x1_class_0, x2_class_0], np.c_[x1_class_1, x2_class_1]))
y = np.array([0] * (num_samples // 2) + [1] * (num_samples // 2))  # Labels
x = np.c_[np.ones(x.shape[0]), x]  # Add bias term

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Logistic regression training
def train_logistic_regression(x, y, lr=0.01, epochs=1000):
    weights = np.zeros(x.shape[1])  # Initialize weights
    for _ in range(epochs):
        z = np.dot(x, weights)
        predictions = sigmoid(z)
        gradient = np.dot(x.T, (predictions - y)) / len(y)
        weights -= lr * gradient
    return weights

# Prediction function
def predict(x, weights):
    return (sigmoid(np.dot(x, weights)) >= 0.5).astype(int)

# Metrics
def compute_confusion_matrix(y_true, y_pred):
    tn = np.sum((y_true == 0) & (y_pred == 0))
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return np.array([[tn, fp], [fn, tp]])

# Train the model
weights = train_logistic_regression(x, y, lr=0.1, epochs=3000)

# Make predictions
y_pred = predict(x, weights)

# Evaluate the model
conf_matrix = compute_confusion_matrix(y, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
accuracy = (tp + tn) / len(y)
precision = tp / (tp + fp) if tp + fp > 0 else 0
recall = tp / (tp + fn) if tp + fn > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

# Print results
print("Confusion Matrix:")
print(f"       Predicted 0   Predicted 1")
print(f"Actual 0      {conf_matrix[0, 0]}             {conf_matrix[0, 1]}")
print(f"Actual 1      {conf_matrix[1, 0]}             {conf_matrix[1, 1]}")
print(f"\nAccuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1_score:.2f}")

# Plot the decision boundary
x1_min, x1_max = x[:, 1].min() - 1, x[:, 1].max() + 1
x2_min, x2_max = x[:, 2].min() - 1, x[:, 2].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100))
grid = np.c_[np.ones(xx1.ravel().shape), xx1.ravel(), xx2.ravel()]
probs = sigmoid(np.dot(grid, weights)).reshape(xx1.shape)

plt.contourf(xx1, xx2, probs, levels=[0, 0.5, 1], alpha=0.3, colors=['blue', 'red'])
plt.scatter(x[:, 1], x[:, 2], c=y, cmap='bwr', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()
