import cv2
import matplotlib.pyplot as plt

def pooling(image, pool_size=2):
    """Perform different pooling operations"""
    # Convert to grayscale if needed
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
    
    # Perform pooling
    max_pool = cv2.resize(gray, (gray.shape[1]//pool_size, gray.shape[0]//pool_size), 
                           interpolation=cv2.INTER_NEAREST)
    min_pool = cv2.resize(gray, (gray.shape[1]//pool_size, gray.shape[0]//pool_size), 
                           interpolation=cv2.INTER_NEAREST)
    avg_pool = cv2.resize(gray, (gray.shape[1]//pool_size, gray.shape[0]//pool_size), 
                          interpolation=cv2.INTER_AREA)
    
    return max_pool, min_pool, avg_pool

def main():
    # Get image path from user
    image_path = input("Enter image path: ")
    
    try:
        # Read image
        image = cv2.imread(image_path)
        
        # Perform pooling
        max_pool, min_pool, avg_pool = pooling(image)
        
        # Convert original image to RGB for display
        original_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Visualize results
        plt.figure(figsize=(12, 3))
        titles = ['Original', 'Max Pooling', 'Min Pooling', 'Average Pooling']
        images = [original_rgb, max_pool, min_pool, avg_pool]
        
        for i, (title, img) in enumerate(zip(titles, images), 1):
            plt.subplot(1, 4, i)
            plt.title(title)
            plt.imshow(img, cmap='gray' if i > 1 else None)
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    except Exception as e:
        print(f"Error processing image: {e}")

if __name__ == "__main__":
    main()