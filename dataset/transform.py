import numpy as np, torch, cv2

ALPHA = 0.4
SIZE = 512
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PALETTE = np.array([[0, 0, 0], [34, 139, 34], [205, 133, 63], [255, 215, 0], [0, 191, 255]], dtype=np.uint8)


def preprocess(image, device):
    '''
        Preparazione di una immagine (letta con OpenCV) in un tensore
    '''
    image = cv2.resize(image, (SIZE, SIZE))
    image = (image.astype(np.float32) / 255.0)
    image = (image - MEAN) / STD        # -- Normalizzazione
    image = np.transpose(image, (2, 0, 1))
    tensor = torch.tensor(image).float().unsqueeze(0).to(device)
    return tensor

def postprocess(output, original, alpha=ALPHA):
    '''
        Dall'output del modello, estrae la maschera, applica la palette e ricostruisce l'overlay
    '''
    mask = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    mask_color = PALETTE[mask]
    
    # Riporto della maschera colorata alla risoluzione originaria
    mask_color = cv2.resize(mask_color, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Formula: output = (original * (1 - alpha)) + (mask_color * alpha) + 0
    overlay = cv2.addWeighted(original, 1 - alpha, mask_color, alpha, 0)
    return overlay.astype(np.uint8)

def count_parameters(model):
    '''
        Numero totale di parametri nel modello
    '''

    # numel(): numero di elementi totali all'interno di ogni singolo tensore dei parametri
    return sum(p.numel() for p in model.parameters())

