import numpy as np
import matplotlib.pyplot as plt

class LeNet:
    def __init__(self, input_shape=(8, 8), num_classes=10):
        # Set random seed for reproducibility
        np.random.seed(42)
        
        # Layer dimensions
        self.input_shape = input_shape
        self.num_classes = num_classes
        
        # Convolutional Layer 1 weights and biases
        self.W1 = np.random.randn(6, 1, 5, 5) * 0.1
        self.b1 = np.zeros((6, 1))
        
        # Convolutional Layer 2 weights and biases
        self.W2 = np.random.randn(16, 6, 5, 5) * 0.1
        self.b2 = np.zeros((16, 1))
        
        # Fully Connected Layer weights and biases
        self.W3 = np.random.randn(120, 16) * 0.1
        self.b3 = np.zeros((120, 1))
        
        self.W4 = np.random.randn(84, 120) * 0.1
        self.b4 = np.zeros((84, 1))
        
        self.W5 = np.random.randn(num_classes, 84) * 0.1
        self.b5 = np.zeros((num_classes, 1))
    
    def conv2d(self, X, W, b):
        # Simple convolution implementation
        batch_size, in_channels, in_height, in_width = X.shape
        num_filters, _, filter_height, filter_width = W.shape
        
        out_height = in_height - filter_height + 1
        out_width = in_width - filter_width + 1
        
        output = np.zeros((batch_size, num_filters, out_height, out_width))
        
        for n in range(batch_size):
            for f in range(num_filters):
                for h in range(out_height):
                    for w in range(out_width):
                        output[n, f, h, w] = np.sum(
                            X[n, :, h:h+filter_height, w:w+filter_width] * W[f]) + b[f]
        
        return output
    
    def max_pool(self, X):
        # Max pooling with 2x2 window
        batch_size, channels, height, width = X.shape
        output = np.zeros((batch_size, channels, height//2, width//2))
        
        for n in range(batch_size):
            for c in range(channels):
                for h in range(0, height, 2):
                    for w in range(0, width, 2):
                        output[n, c, h//2, w//2] = np.max(X[n, c, h:h+2, w:w+2])
        
        return output
    
    def softmax(self, X):
        # Softmax activation
        exp_X = np.exp(X - np.max(X, axis=1, keepdims=True))
        return exp_X / np.sum(exp_X, axis=1, keepdims=True)
    
    def forward(self, X):
        # Reshape input
        X = X.reshape(-1, 1, *self.input_shape)
        
        # Convolution Layer 1 with ReLU
        conv1 = np.maximum(0, self.conv2d(X, self.W1, self.b1))
        pool1 = self.max_pool(conv1)
        
        # Convolution Layer 2 with ReLU
        conv2 = np.maximum(0, self.conv2d(pool1, self.W2, self.b2))
        pool2 = self.max_pool(conv2)
        
        # Flatten
        flattened = pool2.reshape(pool2.shape[0], -1)
        
        # Fully Connected Layers with ReLU
        fc1 = np.maximum(0, np.dot(flattened, self.W3.T) + self.b3.T)
        fc2 = np.maximum(0, np.dot(fc1, self.W4.T) + self.b4.T)
        
        # Output Layer with Softmax
        output = self.softmax(np.dot(fc2, self.W5.T) + self.b5.T)
        
        return output
    
    def train(self, X, y, epochs=100, learning_rate=0.01):
        # Training history
        train_accuracies = []
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.forward(X)
            
            # Calculate accuracy
            predicted_classes = np.argmax(predictions, axis=1)
            accuracy = np.mean(predicted_classes == y)
            train_accuracies.append(accuracy)
            
            # Print progress
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Accuracy: {accuracy*100:.2f}%")
        
        return train_accuracies

def generate_digits_dataset():
    """
    Generate a simple synthetic digits dataset
    Similar to the sklearn digits dataset structure
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Generate synthetic data
    num_samples = 1000
    input_shape = (8, 8)
    num_classes = 10
    
    # Create input data
    X = np.random.randn(num_samples, *input_shape)
    
    # Create labels with some structure
    y = np.zeros(num_samples, dtype=int)
    for i in range(num_samples):
        y[i] = i % num_classes
    
    # Normalize data
    X = (X - X.mean()) / X.std()
    
    return X, y

def split_data(X, y, test_size=0.2):
    """
    Manually split data into train and test sets
    """
    np.random.seed(42)
    
    # Shuffle the data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Calculate split index
    split_idx = int(len(X) * (1 - test_size))
    
    # Split data
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test

def main():
    # Generate synthetic digits dataset
    X, y = generate_digits_dataset()
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Create and train LeNet
    lenet = LeNet()
    train_accuracies = lenet.train(X_train, y_train)
    
    # Plotting accuracy over epochs
    plt.figure(figsize=(10, 5))
    plt.plot(train_accuracies)
    plt.title('Training Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()