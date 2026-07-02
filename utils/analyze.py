import random, cv2, os
import matplotlib.pyplot as plt
import numpy as np

MASKS = './car-segmentation/masks'
CLASSES = ['Background', 'Car', 'Wheels', 'Lights', 'Windows']
PALETTE = np.array([[0,0,0], [34,139,34], [205,133,63], [255,215,0], [0,191,255]], dtype=np.uint8)
SAMPLES = 2

def set_mask(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}.")
    return [os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith(".png")]

def get_mask(path):
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Impossible to load: {path}.")
    return mask

def set_color(mask):
    values = np.unique(mask)
    if np.any(values >= len(PALETTE)):
        raise ValueError(f"Mask with invalid classes: {values}.")
    return PALETTE[mask]

def get_plot(images, counts, files):
    n = len(images)
    fig, axes = plt.subplots(n, 2, figsize=(14, 6 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    class_colors = PALETTE / 255.0

    for i in range(n):
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title(f"Segmentation Mask\n{files[i]}", fontsize=13, fontweight='bold' )
        axes[i, 0].axis("off")

        total_pixels = np.sum(counts[i])
        percentages = (counts[i] / total_pixels) * 100
        x = np.arange(len(CLASSES))

        bars = axes[i, 1].bar(x, percentages, color=class_colors, edgecolor='black', linewidth=1.2, alpha=0.85)
        axes[i, 1].plot(x, percentages, marker='o', linewidth=2,)
        for j, bar in enumerate(bars):
            height = bar.get_height()
            axes[i, 1].text(bar.get_x() + bar.get_width() / 2, height + 0.5, f"{percentages[j]:.1f}%", ha='center', fontsize=9, fontweight='bold')
        axes[i, 1].set_xticks(x)
        axes[i, 1].set_xticklabels(CLASSES, rotation=25)
        axes[i, 1].set_ylabel("Pixel Percentage (%)")
        axes[i, 1].set_title("Class Distribution", fontsize=13, fontweight='bold')

        axes[i, 1].grid(axis='y', linestyle='--', alpha=0.4)
        axes[i, 1].spines['top'].set_visible(False)
        axes[i, 1].spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()

def main():
    files = set_mask(MASKS)
    if len(files) == 0:
        print("Zero file found.")
        return
    
    samples = random.sample(files, min(SAMPLES, len(files)))
    images = []
    counts = []
    names = []
    
    for path in samples:
        original = get_mask(path)
        count = np.bincount(original.flatten(), minlength=len(CLASSES))
        mask = set_color(original)

        images.append(mask)
        counts.append(count)
        names.append(os.path.basename(path))
    get_plot(images, counts, names)

if __name__ == "__main__":
    main()