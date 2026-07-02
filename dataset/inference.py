import matplotlib.pyplot as plt
import torch, random, numpy as np

N = 3
ALPHA = 0.5     # -- Trasparenza dell'overlay tra img e mask
MEAN = np.array([0.485, 0.456, 0.406])
STD = np.array([0.229, 0.224, 0.225])

def get_samples(loader, device, n=N):
    '''
        Estrazione di N campioni dal Data Loader
    '''
    images_all = []
    masks_all = []

    for images, masks in loader:
        for i in range(images.size(0)):
            images_all.append(images[i])
            masks_all.append(masks[i])
            if len(images_all) >= n:
                break
        if len(images_all) >= n:
            break

    if len(images_all) == 0:
        raise ValueError("Loader is empty.")

    # Ricostruzione del batch: unione dei tensori lungo la nuova dim
    images = torch.stack(images_all).to(device)
    masks = torch.stack(masks_all).to(device)
    return images, masks

def set_predict(model, images):
    '''
        Inferenza del modello sulle immagini
    '''
    model.eval()    # -- Modalità valutazione
    with torch.no_grad():
        outputs = model(images)
        preds = outputs.argmax(1)   # -- Indice della classe con la probabilità più alta lungo la dim dei canali
    return preds

def set_denormalize(image):
    '''
        Normalizzazione inversa per visualizzare correttamente l'immagine.
    '''
    # detach(): scollego il tensore dal grafo di computazione
    # permute(): cambio l'ordine degli assi da Pytorch (CHW) a Numpy (HWC)
    image = image.detach().cpu().permute(1, 2, 0).numpy()
    image = image * STD + MEAN      ## Formula inversa: pixel_originale = (pixel_normalizzato * std) + mean
    
    return np.clip(image, 0, 1)     # clip(): arrotondamenti non escano dall'intervallo standard di visualizzazione [0, 1]

def overlay(image, mask, alpha=ALPHA):
    '''
        Sovrapposizione tra originale e maschera di predizione
    '''
    # Se la maschera è a canale singolo (livelli di grigio), la duplica su 3 canali identici per l'interazione RGB
    if mask.ndim == 2:
        mask = np.stack([mask]*3, axis=-1)
    mask = mask.astype(np.float32)
    if mask.max() > 1:
        mask = mask / mask.max()
        
    # Interpolazione lineare pixel per pixel per fondere l'immagine e la maschera colorata
    return np.clip(image * (1 - alpha) + mask * alpha, 0, 1)

def get_predict(images, preds, masks=None, color_fn=None, denorm_fn=None, n=N):
    '''
        Plot Matplotlib con: Immagine -> Mask reale -> Predizione -> Overlay -> Mappa degli errori
    '''
    samples = min(n, len(images), len(preds))
    cols = 5 if masks is not None else 3

    fig, axes = plt.subplots(samples, cols, figsize=(5 * cols, 5 * samples))
    if samples == 1:
        axes = np.expand_dims(axes, axis=0)     # -- Mantiene la consistenza bidimensionale degli assi con un solo campione
        
    for i in range(samples):
        
        # -- Gestione Immagine
        img = images[i]
        if denorm_fn:
            img = denorm_fn(img)
        elif torch.is_tensor(img):
            img = set_denormalize(img)
        
        # -- Gestione Predizione
        pred = preds[i]
        if torch.is_tensor(pred):
            pred = pred.detach().cpu().numpy()  # -- Spostamento su CPU e conversione in array NumPy per Matplotlib
        pred_color = color_fn(pred) if color_fn else pred
    
        # -- Gestione Ground Truth e Calcolo Errore
        if masks is not None:
            gt = masks[i]
            if torch.is_tensor(gt):
                gt = gt.detach().cpu().numpy()
            gt_color = color_fn(gt) if color_fn else gt
            
            # Evidenzia spazialmente dove il modello ha sbagliato la classificazione dei pixel
            error_map = (pred != gt).astype(np.uint8)

        # Rendering della colonna Immagine Originale
        axes[i, 0].imshow(img)
        axes[i, 0].set_title('Image', fontsize=12, fontweight='bold')
        axes[i, 0].axis('off')

        # Rendering della colonna Ground Truth (se disponibile)
        if masks is not None:
            axes[i, 1].imshow(gt_color)
            axes[i, 1].set_title('Ground Truth', fontsize=12, fontweight='bold')
            axes[i, 1].axis('off')

        # Rendering della colonna Predizione del Modello
        pred_col = 2 if masks is not None else 1
        axes[i, pred_col].imshow(pred_color)
        axes[i, pred_col].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[i, pred_col].axis('off')

        # Rendering della colonna Overlay (Trasparenza Immagine + Predizione)
        if masks is not None:
            if pred_color.max() > 1:
                pred_vis = pred_color / 255.0
            else:
                pred_vis = pred_color
            overlay_img = overlay(img, pred_vis)
            axes[i, 3].imshow(overlay_img)
            axes[i, 3].set_title('Overlay', fontsize=12, fontweight='bold')
            axes[i, 3].axis('off')

        # Rendering della colonna Mappa Errori e calcolo percentuale di errore sul totale dei pixel
        if masks is not None:
            axes[i, 4].imshow(error_map, cmap='Reds')   # -- Usa una mappa di colore rossa per enfatizzare gli errori
            error_pct = (error_map.mean() * 100)        # -- La media dei pixel binari rappresenta la percentuale di errore macroscopica
            axes[i, 4].set_title(f'Errors\n{error_pct:.2f}%', fontsize=12, fontweight='bold')
            axes[i, 4].axis('off')

    plt.tight_layout()
    plt.show()

