import matplotlib.pyplot as plt
import torch, numpy as np

CLASSES = 5
EPS = 1e-6
DELTA = 1e-4

def train(model, loader, optimizer, criterion, device, classes=CLASSES):
    '''
        Una epoca di training
    '''
    model.train()
    total_loss = 0.0
    confusion = torch.zeros((classes, classes), device=device)

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()       # -- Ripristino i gradienti
        outputs = model(images)     # -- Forward pass: calcolo dei logit predetti dal modello
        
        loss = criterion(outputs, masks)
        loss.backward()             # -- Backward pass: calcola i gradienti della loss rispetto ai parametri del modello tramite backpropagation
        
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(1)
        conf = confusion_matrix(preds, masks, device=device)
        confusion += conf
    
    # Metriche medie globali sull'intera epoca
    iou = get_iou(confusion).mean().item()
    dice = get_dice(confusion).mean().item()
    return total_loss/len(loader), iou, dice

def validation(model, loader, criterion, device, classes=CLASSES):
    '''
        Validazione del modello
    '''
    model.eval()
    total_loss = 0.0
    confusion = torch.zeros((classes, classes), device=device)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            preds = outputs.argmax(1)
            conf = confusion_matrix(preds, masks, device=device)
            confusion += conf
    
    iou = get_iou(confusion).mean().item()
    dice = get_dice(confusion).mean().item()
    return total_loss/len(loader), iou, dice

def set_early_stopping(model, model_path, current_value, best_value, pat_count, patience, mode='min', min_delta=DELTA, verbose=True):
    '''
        Early Stopping: interruzione del training se la metrica monitorata smette di migliorare
    '''
    stop = False
    # Check: controllo il miglioramento considerando 'min_delta' (non azzera la pazienza per fluttuazioni microscopiche)
    if mode == 'min':
        improved = current_value < best_value - min_delta  # -- Usato tipicamente per la Loss
    else:
        improved = current_value > best_value + min_delta  # -- Usato tipicamente per IoU o Dice

    if improved:
        best_value = current_value
        pat_count = 0           # -- Ripristina il contatore della pazienza
        torch.save(model.state_dict(), model_path)
        if verbose:
            print(f"[WARNING] Best model saved. Path: {model_path}.")
    else:
        pat_count += 1          # -- Incrementa se non c'è stato alcun miglioramento significativo
        if verbose:
            print(f"[WARNING] Patience: {pat_count}/{patience}.")
        if pat_count >= patience:
            if verbose:
                print(f"[WARNING] Early stopping: active.")
            stop = True         # -- Segnala che l'addestramento globale deve interrompersi
            
    return best_value, pat_count, stop

def smooth(values, weight=0.6):
    '''
        Smorzamento delle fluttuazioni dei grafici, per avere trend leggibili
    '''
    if len(values) == 0:
        return values
    smoothed = []
    last = values[0]
    for v in values:
        s = weight * last + (1 - weight) * v
        smoothed.append(s)
        last = s
    return smoothed

def get_plot(history: dict):
    '''
        Dashboard di monitoraggio: Loss, IoU, Dice, Overfitting GAP
    '''
    epochs = np.arange(1, len(history['train_loss']) + 1)

    # Smoothing prima del plot per pulire il rumore visivo
    train_loss = smooth(history['train_loss'])
    val_loss = smooth(history['val_loss'])
    train_iou = smooth(history['train_iou'])
    val_iou = smooth(history['val_iou'])
    train_dice = smooth(history['train_dice'])
    val_dice = smooth(history['val_dice'])
    
    # Epoca in cui si è ottenuto il picco di validazione
    best_epoch = np.argmax(history['val_iou']) + 1
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle("Training dashboard", fontsize=18, fontweight="bold")

    # -- Loss Plot
    ax = axes[0, 0]
    ax.plot(epochs, train_loss, linewidth=3, label='Train Loss')
    ax.plot(epochs, val_loss, linewidth=3, label='Validation Loss')
    ax.axvline(best_epoch, linestyle='--', alpha=0.7)
    ax.scatter(best_epoch, history['val_loss'][best_epoch - 1], s=100, zorder=5)
    ax.set_title('Loss', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid(linestyle='--', alpha=0.3)
    ax.legend()

    # -- IoU Plot
    ax = axes[0, 1]
    ax.plot(epochs, train_iou, linewidth=3, label='Train IoU')
    ax.plot(epochs, val_iou, linewidth=3, label='Validation IoU')
    ax.axvline(best_epoch, linestyle='--', alpha=0.7)
    ax.set_title('Mean IoU', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU')
    ax.grid(linestyle='--', alpha=0.3)
    ax.legend()

    # -- Dice Score Plot
    ax = axes[1, 0]
    ax.plot(epochs, train_dice, linewidth=3, label='Train Dice')
    ax.plot(epochs, val_dice, linewidth=3, label='Validation Dice')
    ax.axvline(best_epoch, linestyle='--', alpha=0.7)
    ax.set_title('Dice Score', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Dice')
    ax.grid(linestyle='--', alpha=0.3)
    ax.legend()

    # -- Overfitting Gap Plot
    # Calcolo della divergenza numerica tra l'accuratezza su train e su validation
    ax = axes[1, 1]
    gap = np.array(history['train_iou']) - np.array(history['val_iou'])
    ax.plot(epochs, gap, linewidth=3)
    ax.axhline(0, linestyle='--', alpha=0.5)
    ax.set_title('Overfitting Gap (IoU)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Train IoU - Val IoU')
    ax.grid(linestyle='--', alpha=0.3)

    best_iou = max(history['val_iou'])
    best_dice = max(history['val_dice'])
    text = (f"Best epoch: {best_epoch}\nBest Val IoU: {best_iou:.4f}\nBest Val Dice score: {best_dice:.4f}.")
    fig.text(0.85, 0.02, text, fontsize=11, bbox=dict(facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.show()
    
def confusion_matrix(preds, targets, classes=CLASSES, device=None):
    '''
        Matrice di Confusione (algoritmo vettorizzato)
    '''
    # Appiattisce le predizioni e i target in tensori monodimensionali (vettori di pixel)
    preds = preds.view(-1)
    targets = targets.view(-1)

    mask = (targets >= 0) & (targets < classes) # -- Maschera booleana per considerare solo i pixel con target validi
    
    # Codifica delle combinazioni uniche (target, pred) in un unico indice lineare
    # torch.bincount per contare istantaneamente le frequenze delle occorrenze, rimodellandole in una matrice quadrata
    cm = torch.bincount(classes * targets[mask] + preds[mask], minlength=classes**2).reshape(classes, classes)
    if device is not None:
        cm = cm.to(device)
    return cm

def get_iou(cm, eps=EPS):
    '''
        Intersect over Union
    '''
    intersection = torch.diag(cm)                   # -- La diagonale contiene i True Positives
    union = cm.sum(1) + cm.sum(0) - intersection    # --  Unione = Somma Riga (Predetti) + Somma Colonna (Reali) - True Positives (contati due volte)
    return (intersection + eps) / (union + eps)

def get_dice(cm, eps=EPS):
    '''
        Dice Score
    '''
    intersection = torch.diag(cm)
    denom = cm.sum(1) + cm.sum(0)
    return (2 * intersection + eps) / (denom + eps)
