import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

def apply_max_pooling(image, pool_size=2):
    """
    Apply max pooling to the input image
    
    Args:
    image (numpy.ndarray): Input grayscale image
    pool_size (int): Size of the pooling window
    
    Returns:
    numpy.ndarray: Max pooled image
    """
    height, width = image.shape
    output_height = height // pool_size
    output_width = width // pool_size
    
    max_pooled = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            window = image[
                i*pool_size : (i+1)*pool_size, 
                j*pool_size : (j+1)*pool_size
            ]
            max_pooled[i, j] = np.max(window)
    
    return max_pooled

def apply_min_pooling(image, pool_size=2):
    """
    Apply min pooling to the input image
    
    Args:
    image (numpy.ndarray): Input grayscale image
    pool_size (int): Size of the pooling window
    
    Returns:
    numpy.ndarray: Min pooled image
    """
    height, width = image.shape
    output_height = height // pool_size
    output_width = width // pool_size
    
    min_pooled = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            window = image[
                i*pool_size : (i+1)*pool_size, 
                j*pool_size : (j+1)*pool_size
            ]
            min_pooled[i, j] = np.min(window)
    
    return min_pooled

def apply_average_pooling(image, pool_size=2):
    """
    Apply average pooling to the input image
    
    Args:
    image (numpy.ndarray): Input grayscale image
    pool_size (int): Size of the pooling window
    
    Returns:
    numpy.ndarray: Average pooled image
    """
    height, width = image.shape
    output_height = height // pool_size
    output_width = width // pool_size
    
    avg_pooled = np.zeros((output_height, output_width))
    
    for i in range(output_height):
        for j in range(output_width):
            window = image[
                i*pool_size : (i+1)*pool_size, 
                j*pool_size : (j+1)*pool_size
            ]
            avg_pooled[i, j] = np.mean(window)
    
    return avg_pooled

def main():
    # Create a sample grayscale image
    # Using a gradient image to clearly show pooling effects
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    image = X + Y
    
    # Apply pooling operations
    max_pooled = apply_max_pooling(image)
    min_pooled = apply_min_pooling(image)
    avg_pooled = apply_average_pooling(image)
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(image, cmap=cm.gray)
    plt.axis('off')
    
    plt.subplot(142)
    plt.title('Max Pooling')
    plt.imshow(max_pooled, cmap=cm.gray)
    plt.axis('off')
    
    plt.subplot(143)
    plt.title('Min Pooling')
    plt.imshow(min_pooled, cmap=cm.gray)
    plt.axis('off')
    
    plt.subplot(144)
    plt.title('Average Pooling')
    plt.imshow(avg_pooled, cmap=cm.gray)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()