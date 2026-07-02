from dataset.train import confusion_matrix, get_iou, get_dice
import torch

CLASSES = 5

def get_model(model_class, path, device, verbose=True, **kwargs):
    '''
        Inizializzazione del modello: carico i pesi, preparazione per il test/val
    '''
    model = model_class(**kwargs).to(device)        # -- Inizializza dinamicamente l'architettura passando eventuali parametri extra (**kwargs) e la sposta sul device
    
    model.load_state_dict(torch.load(path, map_location=device))    # -- 'map_location=device' assicura che i pesi vengano caricati direttamente sul dispositivo corrente (es. CUDA o CPU)
    model.eval()
    
    if verbose:
        print(f"[WARNING] Model load, path: {path}, device: {device}.")
    return model

def evaluate(model, loader, device, classes=CLASSES):
    '''
        Valutazione del modello sul Data Loader producendo la matrice di confusione pixel by pixel
    '''
    model.eval()
    
    # CRUCIALE: Inizializza la matrice di confusione (Classi x Classi) sul device
    confusion = torch.zeros((classes, classes), device=device)  # -- Accumulare i conteggi dei pixel True Positive, False Positive e False Negative
    
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)         # -- Forward pass: calcolo dei logit/probabilità da parte della rete
            preds = outputs.argmax(1)       # -- Converte le probabilità nell'indice della classe predetta (argomenti massimi lungo l'asse dei canali)
            
            # Calcola la matrice di confusione per il batch corrente confrontando ogni singolo pixel predetto con il ground truth
            cm = confusion_matrix(preds, masks).to(device)
            confusion += cm                 # -- Accumula i risultati nel tensore globale

    # Calcolo delle metriche di segmentazione sfruttando la matrice di confusione complessiva
    iou = get_iou(confusion)
    dice = get_dice(confusion)
    
    return {
        'iou_class': iou.detach().cpu().numpy(),
        'dice_class': dice.detach().cpu().numpy(),
        'iou': iou.mean().item(),
        'dice': dice.mean().item()
    }
