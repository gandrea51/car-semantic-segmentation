from torch.utils.data import Dataset, DataLoader, Subset, random_split
from albumentations.pytorch import ToTensorV2
import albumentations as A
import torch, cv2, os
import numpy as np

IMAGES = './car-segmentation/images'
MASKS = './car-segmentation/masks'
CLASSES = 5
SIZE = 512
BATCH = 4
WORKERS = 2
SEED = 62
MEAN = (0.485, 0.456, 0.406)    # -- Normalizzazione di IMAGENET
STD = (0.229, 0.224, 0.225)

def data_augmentation():
    '''
        Pipeline di Data Augmentation (albumentations: trasformazioni applicate sia all'immagine che alla maschera)
    '''
    train_tf = A.Compose([
        A.HorizontalFlip(p=0.5),
        # Affine gestisce rotazione, scala e traslazione mantenendo i bordi costanti per la maschera
        A.Affine(scale=(0.85, 1.15), translate_percent=0.05, rotate=(-10, 10), border_mode=cv2.BORDER_CONSTANT, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.GaussNoise(std_range=(0.02, 0.05), p=0.2),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=MEAN, std=STD),
        A.ToTensorV2()
    ])
    
    val_tf = A.Compose([
        A.Resize(SIZE, SIZE),
        A.Normalize(mean=MEAN, std=STD),
        A.ToTensorV2()
    ])
    return train_tf, val_tf

class CarDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        super().__init__()
        self.transform = transform
        self.samples = []

        # Verifica preliminare delle coppie (img, mask)
        files = sorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        for file in files:
            image_path = os.path.join(image_dir, file)
            mask_path = os.path.join(mask_dir, file)
            # Check: manca una maschera corrispondente
            if not os.path.exists(mask_path):
                print(f"[WARNING] Missing mask: {file}.")
                continue
            self.samples.append((image_path, mask_path))
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid image-mask pairs found.")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        
        image = cv2.imread(image_path)      # -- Lettura in BGR
        if image is None:
            raise ValueError(f"Error loading image: {image_path}.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)      # -- Converto in RGB

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)  # -- Mask in scala di grigi: pixel = classe
        if mask is None:
            raise ValueError(f"Error loading mask: {mask_path}.")
        
        # Check: indice di classe nella mask > classi attese
        unique = np.unique(mask)
        if np.any(unique >= CLASSES):
            raise ValueError(f"Invalid mask classes in {mask_path}: {unique}.")
        
        # Data Augmentation: img, mask
        if self.transform:
            augmentation = self.transform(image=image, mask=mask)
            image = augmentation['image']
            mask = augmentation['mask'].long()      # -- Mask di tipo long
            
        return image, mask


def dataloaders(image_dir=IMAGES, mask_dir=MASKS, batch=BATCH, workers=WORKERS, seed=SEED):
    train_tf, test_tf = data_augmentation()
    
    dataset = CarDataset(image_dir, mask_dir)   # -- Dataset iniziale per il calcolo della divisione degli indici
    
    # Proporzioni: 70% Train, 15% Val, 15% Test
    n = len(dataset)
    train_size = int(0.7 * n)
    val_size = int(0.15 * n)
    test_size = n - train_size - val_size
    
    # Generatore con seed fisso per riproducibilità
    generator = torch.Generator().manual_seed(seed)
    train_subset, val_subset, test_subset = random_split(dataset, [train_size, val_size, test_size], generator=generator)

    # Sotto-dataset finali
    train_dataset = Subset(CarDataset(image_dir, mask_dir, transform=train_tf), train_subset.indices)
    val_dataset = Subset(CarDataset(image_dir, mask_dir, transform=test_tf), val_subset.indices)
    test_dataset = Subset(CarDataset(image_dir, mask_dir, transform=test_tf), test_subset.indices)

    # Data Loader - shuffle solo nel training (evitare che venga memorizzato l'ordine dei dati)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch, shuffle=False, num_workers=workers)
    
    return train_loader, val_loader, test_loader

