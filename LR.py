import numpy as np
import matplotlib.pyplot as plt

def linear_regression(x, y):
    # Convert input to NumPy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Calculate means
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    
    # Calculate slope and intercept
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    return slope, intercept

# Input data
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [3, 4, 5, 6, 8, 9, 11, 13, 14, 5]

# Perform linear regression
slope, intercept = linear_regression(x, y)

# Predict y values
x = np.array(x)  # Convert x to a NumPy array for calculations
y_pred = intercept + slope * x

# Calculate the sum of squared error (SSE)
sse = np.sum((y - y_pred) ** 2)

# Print results
print("Intercept:", intercept)
print("Slope:", slope)
print(f"Best-fit line equation: y = {slope:.2f}x + {intercept:.2f}")
print(f"Sum of Squared Error (SSE): {sse:.2f}")

# Plot results
plt.scatter(x, y, color='blue', label='Data points')  # Original data points
plt.plot(x, y_pred, color='red', label='Best fit line')  # Best-fit line
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()
