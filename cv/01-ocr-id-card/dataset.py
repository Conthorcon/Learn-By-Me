import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A

IMG_SIZE = 512

# =====================
# DATASET
# =====================
class IDCardDataset(Dataset):
    def __init__(self, img_dir, mask_dir, file_list, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.files = file_list
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]

        img_path = os.path.join(self.img_dir, file)
        mask_path = os.path.join(self.mask_dir, file.replace(".jpg", ".png"))

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(mask_path, 0)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        image = image.transpose(2, 0, 1).astype("float32") / 255.0
        mask = np.expand_dims(mask, axis=0).astype("float32") / 255.0

        return torch.tensor(image), torch.tensor(mask)

# =====================
# AUGMENTATION
# =====================
train_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    # A.HorizontalFlip(p=0.5),
    # A.ShiftScaleRotate(0.1, 0.2, 15, p=0.5),
    A.RandomBrightnessContrast(p=0.5),
])

val_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
])




