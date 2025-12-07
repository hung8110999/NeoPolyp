import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import os

from dataset import PolypDataset, create_kfold_dataset
from unet import UNet
from loss import combined_loss as DiceBCELoss
from transforms import train_transform     
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# CONFIG
# =========================
IMAGE_DIR = "/mnt/d/Data/train/train"
MASK_DIR  = "/mnt/d/Data/train_gt/train_gt"

N_SPLITS = 5
EPOCHS = 30
BATCH_SIZE = 1
LR = 1e-4


def train_one_fold(fold_id, train_dataset, val_dataset):
    print(f"\n======================")
    print(f"   TRAINING FOLD {fold_id}")
    print(f"======================")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

    model = UNet(in_channels=3, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = DiceBCELoss

    best_val_loss = 999

    for epoch in tqdm(range(1, EPOCHS + 1)):
        model.train()
        train_loss = 0

        for imgs, masks in train_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)

            preds = model(imgs)
            loss = criterion(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # --------------------
        # VALIDATION
        # --------------------
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs = imgs.to(device)
                masks = masks.to(device)

                preds = model(imgs)
                loss = criterion(preds, masks)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(f"Fold {fold_id} | Epoch {epoch}/{EPOCHS} — Train Loss: {train_loss:.4f} — Val Loss: {val_loss:.4f}")

        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"checkpoints/best_fold_{fold_id}.pth")


def main():
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")

    folds = create_kfold_dataset(
        image_dir=IMAGE_DIR,
        mask_dir=MASK_DIR,
        n_splits=N_SPLITS
    )

    for fold_id, (train_idx, val_idx, img_paths, mask_paths) in enumerate(folds):

        train_dataset = PolypDataset(
            [img_paths[i] for i in train_idx],
            [mask_paths[i] for i in train_idx],
            transform=train_transform
        )

        val_dataset = PolypDataset(
            [img_paths[i] for i in val_idx],
            [mask_paths[i] for i in val_idx],
            transform=None
        )

        train_one_fold(fold_id + 1, train_dataset, val_dataset)


if __name__ == "__main__":
    main()
