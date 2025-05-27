import os
import argparse
import matplotlib.pyplot as plt
from collections import Counter
from PIL import Image
import numpy as np

def analyze_dataset(data_dir):
    """
    Analyze the dataset and print information about it.
    
    Args:
        data_dir: Path to the dataset directory
    """
    class_dirs = ['front', 'rear', 'left', 'right']
    
    # Count images per class
    class_counts = {}
    image_sizes = []
    aspect_ratios = []
    
    for class_name in class_dirs:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} does not exist")
            continue
            
        images = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        class_counts[class_name] = len(images)
        
        # Sample some images to get size statistics
        sample_size = min(100, len(images))
        sampled_images = np.random.choice(images, sample_size, replace=False)
        
        for img_name in sampled_images:
            try:
                img_path = os.path.join(class_dir, img_name)
                with Image.open(img_path) as img:
                    width, height = img.size
                    image_sizes.append((width, height))
                    aspect_ratios.append(width / height)
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
    
    # Print dataset statistics
    print("\n=== Dataset Statistics ===")
    print(f"Total images: {sum(class_counts.values())}")
    for class_name, count in class_counts.items():
        print(f"{class_name}: {count} images ({count/sum(class_counts.values())*100:.1f}%)")
    
    # Image size statistics
    widths, heights = zip(*image_sizes)
    print("\n=== Image Size Statistics ===")
    print(f"Width - Min: {min(widths)}, Max: {max(widths)}, Mean: {sum(widths)/len(widths):.1f}")
    print(f"Height - Min: {min(heights)}, Max: {max(heights)}, Mean: {sum(heights)/len(heights):.1f}")
    print(f"Aspect Ratio - Min: {min(aspect_ratios):.2f}, Max: {max(aspect_ratios):.2f}, Mean: {sum(aspect_ratios)/len(aspect_ratios):.2f}")
    
    # Plot class distribution
    plt.figure(figsize=(10, 6))
    plt.bar(class_counts.keys(), class_counts.values())
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Number of Images')
    plt.savefig('class_distribution.png')
    
    # Plot image size distribution
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(widths, heights, alpha=0.5)
    plt.title('Image Dimensions')
    plt.xlabel('Width')
    plt.ylabel('Height')
    
    plt.subplot(1, 2, 2)
    plt.hist(aspect_ratios, bins=20, alpha=0.7)
    plt.title('Aspect Ratio Distribution')
    plt.xlabel('Aspect Ratio (width/height)')
    plt.ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('image_size_distribution.png')
    
    print("\nPlots saved as 'class_distribution.png' and 'image_size_distribution.png'")

def main():
    parser = argparse.ArgumentParser(description='Analyze vehicle orientation dataset')
    parser.add_argument('--data_dir', type=str, default='balanced_dataset', help='Path to dataset directory')
    
    args = parser.parse_args()
    analyze_dataset(args.data_dir)

if __name__ == "__main__":
    main() 