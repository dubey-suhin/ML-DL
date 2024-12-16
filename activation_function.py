import numpy as np
import matplotlib.pyplot as plt
  
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
  
def tanh(x):
    return np.tanh(x)
  
def relu(x):
    return np.maximum(0, x)  
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)
  
def softmax(x):
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x, axis=0, keepdims=True)
  
if __name__ == "__main__":
    x = np.linspace(-10, 10, 100)
  
    y_sigmoid = sigmoid(x)
    y_tanh = tanh(x)     
    y_relu = relu(x)
    y_leaky_relu = leaky_relu(x)
    y_softMax = softmax(x)
  
    plt.figure(figsize=(12, 8))
    plt.subplot(3, 2, 1)
    plt.plot(x, y_sigmoid, label="Sigmoid")
    plt.title("Sigmoid")
    plt.grid()
  
    plt.subplot(3, 2, 2)
    plt.plot(x, y_tanh, label="Tanh", color='orange')
    plt.title("Tanh")
    plt.grid()
  
    plt.subplot(3, 2, 3)
    plt.plot(x, y_relu, label="ReLU", color='green')
    plt.title("ReLU")
    plt.grid()
  
    plt.subplot(3, 2, 4)
    plt.plot(x, y_leaky_relu, label="Leaky ReLU", color='red')
    plt.title("Leaky ReLU")
    plt.grid()
  
    plt.subplot(3, 2, 5)     
    plt.plot(x, y_softMax, label="SoftMax", color='yellow')
    plt.title("SoftMax")
    plt.grid()
    plt.tight_layout()
    plt.show()
