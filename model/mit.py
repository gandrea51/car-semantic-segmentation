from dataset.dataset import dataloaders
from dataset.train import train, validation, set_early_stopping, get_plot
from dataset.test import get_model, evaluate
from dataset.inference import get_samples, set_predict, get_predict
from kornia.losses import FocalLoss, DiceLoss
import segmentation_models_pytorch as smp
import torch.nn.functional as F
import numpy as np
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGES = "./car-segmentation/images"
MASKS = "./car-segmentation/masks"
PATH = "./network/mit.pth"
ENCODER = "mit_b2"
DATASET = "imagenet"
WEIGHTS = np.array([0.0458, 0.0772, 0.1674, 0.5289, 0.1806])
CLASSES = 5
EPOCHS = 30
PATIENCE = 5
FREEZE = 8
LR_ENCODER = 1e-4
LR_DECODER = 1e-3
DECAY = 1e-4
HISTORY = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 'train_dice': [], 'val_dice': []}

def mit():
    print(f"\n[START] Using: {DEVICE}.")
    train_ld, val_ld, test_ld = dataloaders(IMAGES, MASKS)
    model = smp.Unet(encoder_name=ENCODER, encoder_weights=DATASET, in_channels=3, classes=CLASSES, activation=None).to(DEVICE)
    for p in model.encoder.parameters():
        p.requires_grad = False

    weights = torch.tensor(WEIGHTS).to(DEVICE)
    focal = FocalLoss(alpha=0.25, gamma=1.5, reduction='mean', weight=weights)
    dice = DiceLoss()
    def criterion(logits, targets):
        probs = F.softmax(logits, dim=1)
        return focal(logits, targets) + dice(probs, targets)
    optimizer = torch.optim.Adam(filter(lambda e: e.requires_grad, model.parameters()), lr=LR_DECODER, weight_decay=DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 4)
    best_iou = 0.0
    patience = 0

    # [Warning] Start train
    for epoch in range(EPOCHS):
        print(f"\n[WARNING] Epoch: {epoch+1}/{EPOCHS}.")
        if epoch == FREEZE:
            print(f"\n[WARNING] Unfreeze encoder.")
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(
                [
                    {'params': model.encoder.parameters(), 'lr': LR_ENCODER},
                    {'params': model.decoder.parameters(), 'lr': LR_DECODER},
                    {'params': model.segmentation_head.parameters(), 'lr': LR_DECODER},
                ], weight_decay=DECAY            
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 0.5, 4)

        train_loss, train_iou, train_dice = train(model, train_ld, optimizer, criterion, DEVICE)
        val_loss, val_iou, val_dice = validation(model, val_ld, criterion, DEVICE)
        scheduler.step(val_loss)
        HISTORY['train_loss'].append(train_loss)
        HISTORY['val_loss'].append(val_loss)
        HISTORY['train_iou'].append(train_iou)
        HISTORY['val_iou'].append(val_iou)
        HISTORY['train_dice'].append(train_dice)
        HISTORY['val_dice'].append(val_dice)
        print(f"\n[TRAIN RESULT] Loss = {train_loss:.4f}, IoU = {train_iou:.4f}, Dice = {train_dice:.4f}.")
        print(f"\n[VAL RESULT] Loss = {val_loss:.4f}, IoU = {val_iou:.4f}, Dice = {val_dice:.4f}.")
        
        best_iou, patience, stop = set_early_stopping(model, PATH, val_iou, best_iou, patience, PATIENCE, 'max')
        if stop: 
            break
    
    # [WARNING] Stop train
    print(f"\n[END TRAIN] Model saved: ({PATH}).")
    get_plot(HISTORY)

    print(f"\n[START TEST] Test.")
    model = get_model(smp.Unet, PATH, DEVICE, encoder_name=ENCODER, encoder_weights=None, in_channels=3, classes=CLASSES)

    result = evaluate(model, test_ld, DEVICE, CLASSES)
    print(
        f"\n[WARNING] IoU per classe: {result['iou_class']}."
        f"\n[WARNING] IoU medio: {result['iou']}."
        f"\n[WARNING] Dice per classe: {result['dice_class']}."
        f"\n[WARNING] Dice medio: {result['dice']}."
    )

    print(f"\n[END TEST] Test.")
    imgs, msks = get_samples(test_ld, DEVICE)
    preds = set_predict(model, imgs)
    get_predict(imgs, preds, msks)

if __name__ == "__main__":
    mit()