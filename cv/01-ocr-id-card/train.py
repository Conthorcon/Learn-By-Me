import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from dataset import IDCardDataset, train_transform, val_transform
from torch.utils.data import DataLoader

import os

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from metric import dice_score, iou_score

# =====================
# CONFIG
# =====================
BATCH_SIZE = 8
EPOCHS = 20
LR = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

IMG_DIR = "/home/truongkhanh/work/learn-by-me/cv/01-ocr-id-card/dataset/ABC/train_frames/image"
MASK_DIR = "/home/truongkhanh/work/learn-by-me/cv/01-ocr-id-card/dataset/ABC/train_masks/image"

def train(model, optimizer, losses, data):
    dice_loss, bce_loss = losses
    train_loader, val_loader = data
    best_iou = 0

    for epoch in range(EPOCHS):
        # ===== TRAIN =====
        model.train()
        train_loss = 0

        for images, masks in tqdm(train_loader, desc=f'Epoch {epoch}'):
            images = images.to(DEVICE)
            masks = masks.to(DEVICE)

            preds = model(images)

            loss = dice_loss(preds, masks) + bce_loss(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()


        # ===== VALIDATION =====
        model.eval()
        val_loss = 0
        total_iou = 0
        total_dice = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(DEVICE)
                masks = masks.to(DEVICE)

                preds = model(images)

                loss = dice_loss(preds, masks) + bce_loss(preds, masks)
                val_loss += loss.item()

                probs = torch.sigmoid(preds)

                total_iou += iou_score(probs, masks).item()
                total_dice += dice_score(probs, masks).item()

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)
        avg_iou        = total_iou / len(val_loader)
        avg_dice       = total_dice / len(val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss:   {avg_val_loss:.4f}")
        print(f"IoU:        {avg_iou:.4f}")
        print(f"Dice:       {avg_dice:.4f}")

        # ===== SAVE BEST =====
        if avg_iou > best_iou:
            best_iou = avg_iou
            torch.save(model.state_dict(), "best_model.pth")
            print("🔥 Saved Best Model!")

    print("Training Done!")

def main():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=1,
    ).to(DEVICE)

    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss  = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    losses = (dice_loss, bce_loss)

    # =====================
    # DATA SPLIT
    # =====================
    files = os.listdir(IMG_DIR)
    train_files, val_files = train_test_split(files, test_size=0.2, random_state=42)

    train_dataset = IDCardDataset(IMG_DIR, MASK_DIR, train_files, train_transform)
    val_dataset   = IDCardDataset(IMG_DIR, MASK_DIR, val_files, val_transform)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    data = (train_loader, val_loader)

    train(model, optimizer, losses, data)

if __name__ == '__main__':
    main()
