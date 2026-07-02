from dataset.test import get_model
import segmentation_models_pytorch as smp
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
import torch, cv2

IMAGE = './images/cars/capture.png'

UNET_PATH = "./network/mit.pth"
DEEPLAB_PATH = "./network/deeplab.pth"
UNET_ENCODER = "mit_b2"
DEEPLAB_ENCODER = "resnet18"
CLASSES = ['Background', 'Car', 'Wheels', 'Lights', 'Windows']
PALETTE = np.array([[0,0,0], [34,139,34], [205,133,63], [255,215,0], [0,191,255]], dtype=np.uint8)
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SIZE = 512

def preprocess(image):
    image = cv2.resize(image, (SIZE, SIZE))
    image = image.astype(np.float32) / 255.0
    image = (image - MEAN) / STD
    image = np.transpose(image, (2, 0, 1))
    return torch.tensor(image).unsqueeze(0)

def denormalize(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    image = STD * image + MEAN
    return np.clip(image, 0, 1)

def colorize(mask):
    return PALETTE[mask]

def predict(model, image):
    with torch.no_grad():
        logits = model(image.to(DEVICE))
        probs = F.softmax(logits, dim=1)

        preds = torch.argmax(probs, dim=1)
        conf = torch.max(probs, dim=1).values
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
    return (
        preds.squeeze(0).cpu().numpy(),
        conf.squeeze(0).cpu().numpy(),
        entropy.squeeze(0).cpu().numpy()
    )
    
def overlay(image, mask, alpha=0.5):
    return np.clip(image * (1-alpha) + mask / 255.0 * alpha, 0, 1)

def dashboard(image, mask_pred, conf_map, entropy_map):
    denorm = denormalize(image.squeeze(0))
    mask_color = colorize(mask_pred)
    over = overlay(denorm, mask_color)

    cols = 4
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 10))

    # Entropy
    im0 = axes[0, 0].imshow(entropy_map, cmap="magma")
    axes[0, 0].set_title("Entropy (uncertainty)")
    axes[0, 0].axis("off")
    fig.colorbar(im0, ax=axes[0, 0])

    # Original
    axes[0, 1].imshow(denorm)
    axes[0, 1].set_title("Original")
    axes[0, 1].axis("off")

    # Overlap
    axes[0, 2].imshow(over)
    axes[0, 2].set_title("Overlay")
    axes[0, 2].axis("off")

    # Confidence
    im1 = axes[0, 3].imshow(conf_map, cmap="inferno")
    axes[0, 3].set_title("Confidence")
    axes[0, 3].axis("off")
    fig.colorbar(im1, ax=axes[0, 3])

    # Classes
    for c in range(1, len(CLASSES)):
        mask = (mask_pred == c)

        class_img = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
        class_img[mask] = PALETTE[c]

        ax = axes[1, c - 1]
        ax.imshow(class_img)
        ax.set_title(CLASSES[c])
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def main():
    image = preprocess(cv2.cvtColor(cv2.imread(IMAGE), cv2.COLOR_BGR2RGB))
    
    # UNET
    unet = get_model(smp.Unet, UNET_PATH, DEVICE, encoder_name=UNET_ENCODER, encoder_weights=None, in_channels=3, classes=len(CLASSES))
    unet.eval(); pred, conf, entropy = predict(unet, image)

    # DEEPLAB
    #deeplab = get_model(smp.DeepLabV3Plus, DEEPLAB_PATH, DEVICE, encoder_name=DEEPLAB_ENCODER, encoder_weights=None, in_channels=3, classes=len(CLASSES))
    #deeplab.eval(); pred, conf, entropy = predict(deeplab, image)
    
    dashboard(image, pred, conf, entropy)

if __name__ == "__main__":
    main()

