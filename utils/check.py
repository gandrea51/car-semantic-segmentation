from utils.analyze import get_mask
import matplotlib.pyplot as plt
import numpy as np
import cv2, os

IMAGES = './car-segmentation/images'
MASKS = './car-segmentation/masks'
CLASSES = ['Background', 'Car', 'Wheels', 'Lights', 'Windows']
PALETTE = np.array([[0,0,0], [34,139,34], [205,133,63], [255,215,0], [0,191,255]], dtype=np.uint8)
N = 5

def get_image(path):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Load error (image): {path}.")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def get_pairs(image_dir, mask_dir):
    files = sorted(os.listdir(image_dir))
    pairs = []
    
    for name in files:
        image = os.path.join(image_dir, name)
        mask = os.path.join(mask_dir, name)
        if os.path.exists(mask):
            pairs.append((image, mask))
    return pairs

def set_color(mask, palette):
    if mask.max() >= len(palette):
        raise ValueError(f"Mask with invalid class. Max found: {mask.max()}.")
    
    h, w = mask.shape
    color = np.zeros((h, w, 3), dtype=np.uint8)
    for i in range(len(palette)):
        color[mask==i] = palette[i]
    return color

def overlay(image, color, alpha=0.5):
    return (image * (1-alpha) + color * alpha).astype(np.uint8)

def get_plot(image, colored, mask, title=""):
    fig = plt.figure(figsize=(14, 6))

    ax1 = plt.subplot(1, 3, 1)
    ax1.imshow(image)
    ax1.set_title("Image")
    ax1.axis("off")

    ax2 = plt.subplot(1, 3, 2)
    ax2.imshow(colored)
    ax2.set_title("Mask")
    ax2.axis("off")

    ax3 = plt.subplot(1, 3, 3)
    ax3.imshow(overlay(image, colored))
    ax3.set_title("Overlay")
    ax3.axis("off")

    if title:
        plt.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()

    counts = np.bincount(mask.flatten(), minlength=len(CLASSES))
    percentages = counts / counts.sum() * 100
    x = np.arange(len(CLASSES))

    plt.figure(figsize=(10, 4))
    bars = plt.bar(x, percentages, alpha=0.8)
    plt.plot(x, percentages, marker='o')

    for i, v in enumerate(percentages):
        plt.text(i, v + 0.5, f"{v:.1f}%", ha='center')

    plt.xticks(x, CLASSES, rotation=20)
    plt.ylabel("Percentage (%)")
    plt.title("Class Distribution")
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.show()

def main(image_dir, mask_dir, X, palette):
    pairs = get_pairs(image_dir, mask_dir)
    if len(pairs) == 0:
        raise ValueError("Zero file found.")
    pairs = pairs[:X]
    
    for i, (image_path, mask_path) in enumerate(pairs):
        image = get_image(image_path)
        mask = get_mask(mask_path)
        colored = set_color(mask, palette)
        get_plot(image, colored, mask, f"Sample {i}")
    
if __name__ == '__main__':
    main(IMAGES, MASKS, N, PALETTE)
