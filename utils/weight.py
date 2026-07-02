from utils.analyze import get_mask
import matplotlib.pyplot as plt
import numpy as np
import cv2, os

MASKS = './car-segmentation/masks'
CLASSES = ['Background', 'Car', 'Wheels', 'Lights', 'Windows']
N = len(CLASSES)

def get_paths(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory not found: {directory}.")
    return sorted(os.path.join(directory, f) for f in os.listdir(directory) if f.lower().endswith('.png'))

def set_counting(mask_path, n):
    total = np.zeros(n, dtype=np.int64)

    for path in mask_path:
        mask = get_mask(path)
        unique = np.unique(mask)
        if np.any(unique >= n):
            raise ValueError(f"Invalid class value in {path}: {unique}.")
        count = np.bincount(mask.flatten(), minlength=n)
        total += count
    return total

def set_frequency(count):
    total = count.sum()
    if total == 0:
        raise ValueError(f"Total pixel count is zero.")
    return count/total

def inverse_frequency(freq, eps=1e-6):
    return 1.0 / (freq + eps)

def log_smoothed(freq):
    return 1.0 / np.log(1.02 + freq)

def median_frequency_balancing(freq, eps=1e-6):
    median = np.median(freq)
    return median / (freq + eps)

def sqrt_inverse(freq, eps=1e-6):
    return 1.0 / np.sqrt(freq + eps)

def effective_number_weights(counts, beta=0.999):
    effective_num = 1.0 - np.power(beta, counts)
    return ((1.0 - beta) / (effective_num + 1e-8))

def set_norm(weights):
    total = weights.sum()
    if total == 0:
        return weights
    return weights/total

def get_weights(count):
    freq = set_frequency(count)
    weigths = {
        'Inverse frequency': set_norm(inverse_frequency(freq)),
        'Log smoothed': set_norm(log_smoothed(freq)),
        'Median frequency': set_norm(median_frequency_balancing(freq)),
        'SQRT inverse': set_norm(sqrt_inverse(freq)),
        'Effective number': set_norm(effective_number_weights(freq))
    }
    return weigths, freq

def get_plots(weights, classes, freq):
    x = np.arange(len(classes))

    # Dataset Distribution
    plt.figure(figsize=(10, 4))
    bars = plt.bar(classes, freq * 100)
    plt.title('Dataset Class Distribution', fontsize=14, fontweight='bold')
    plt.ylabel('Percentage (%)')
    for bar, val in zip(bars, freq):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3, f'{val*100:.2f}%', ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # Line comparison
    plt.figure(figsize=(12, 6))
    for name, w in weights.items():
        plt.plot(x, w, marker='o', linewidth=2, label=name)
    plt.xticks(x, classes, rotation=20)
    plt.ylabel('Normalized Weight')
    plt.title('Weighting Methods Comparison', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Heatmap
    matrix = np.array([w for w in weights.values()])
    plt.figure(figsize=(10, 5))
    im = plt.imshow(matrix, aspect='auto')
    plt.colorbar(im)
    plt.xticks(x, classes, rotation=20)
    plt.yticks(np.arange(len(weights)), list(weights.keys()))
    plt.title('Weight Heatmap', fontsize=14, fontweight='bold')

    # values inside cells
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f'{matrix[i, j]:.2f}', ha='center', va='center', fontsize=8)

    plt.tight_layout()
    plt.show()

def main():
    mask_path = get_paths(MASKS)
    if len(mask_path) == 0:
        print(f"Zero file found.")
        return
    
    count = set_counting(mask_path, N)
    
    print(f"\nClass count")
    print(f"-" * 40)
    for cls, c in zip(CLASSES, count):
        print(f"{cls:<15}: {c}")
    weights, freq = get_weights(count)
    
    print(f"\nWeights")
    print(f"-" * 40)
    for k, v in weights.items():
        print(f"\n{k}")
        for cls, val in zip(CLASSES, v):
            print(f"  {cls:<15}: {val:.4f}")
    
    get_plots(weights, CLASSES, freq)

if __name__ == '__main__':
    main()
