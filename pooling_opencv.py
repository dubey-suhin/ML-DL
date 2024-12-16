import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_sample_image():
    """
    Create a sample grayscale image for demonstration
    """
    # Create a gradient image
    image = np.zeros((300, 300), dtype=np.uint8)
    for i in range(300):
        image[i, :] = i
    return image

def apply_pooling(image, pool_size=2, pooling_type='max'):
    """
    Apply pooling using OpenCV
    
    Args:
    image (numpy.ndarray): Input grayscale image
    pool_size (int): Size of pooling window
    pooling_type (str): Type of pooling ('max', 'min', 'avg')
    
    Returns:
    numpy.ndarray: Pooled image
    """
    # Mapping pooling types to OpenCV constants
    pooling_methods = {
        'max': cv2.REDUCE_MAX,
        'min': cv2.REDUCE_MIN,
        'avg': cv2.REDUCE_AVG
    }
    
    # Apply pooling
    return cv2.resize(
        image, 
        (image.shape[1] // pool_size, image.shape[0] // pool_size), 
        interpolation=pooling_methods.get(pooling_type, cv2.REDUCE_MAX)
    )

def main():
    # Create sample image
    original_image = create_sample_image()
    
    # Apply different pooling techniques
    max_pooled = apply_pooling(original_image, pool_size=2, pooling_type='max')
    min_pooled = apply_pooling(original_image, pool_size=2, pooling_type='min')
    avg_pooled = apply_pooling(original_image, pool_size=2, pooling_type='avg')
    
    # Visualization
    plt.figure(figsize=(15, 5))
    
    # Original image
    plt.subplot(141)
    plt.title('Original Image')
    plt.imshow(original_image, cmap='gray')
    plt.axis('off')
    
    # Max Pooling
    plt.subplot(142)
    plt.title('Max Pooling')
    plt.imshow(max_pooled, cmap='gray')
    plt.axis('off')
    
    # Min Pooling
    plt.subplot(143)
    plt.title('Min Pooling')
    plt.imshow(min_pooled, cmap='gray')
    plt.axis('off')
    
    # Average Pooling
    plt.subplot(144)
    plt.title('Average Pooling')
    plt.imshow(avg_pooled, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()