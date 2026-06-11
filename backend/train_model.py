"""
BGNet Model Trainer
Run: python train_model.py
Generates: bgnet_model.pth

Dataset structure expected:
dataset/
  O+/  → blood smear images
  A+/
  B+/
  AB+/
  O-/
  A-/
  B-/
  AB-/
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import os

CLASSES = ["O+", "A+", "B+", "AB+", "O-", "A-", "B-", "AB-"]
DATASET_PATH = "../dataset"
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class BGNet(nn.Module):
    def __init__(self, num_classes=8):
        super(BGNet, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


def train():
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    # Dataset
    if not os.path.exists(DATASET_PATH):
        print(f"Dataset not found at {DATASET_PATH}")
        print("Creating demo model with random weights...")
        model = BGNet()
        torch.save(model.state_dict(), "bgnet_model.pth")
        print("Saved bgnet_model.pth (demo weights)")
        return

    full_dataset = ImageFolder(DATASET_PATH, transform=train_transform)
    train_size = int(0.85 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    val_ds.dataset.transform = val_transform

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"Training: {len(train_ds)} | Validation: {len(val_ds)}")
    print(f"Classes: {full_dataset.classes}")

    # Model
    model = BGNet(num_classes=8).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

    best_acc = 0.0

    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validate
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                out = model(imgs)
                _, pred = torch.max(out, 1)
                correct += (pred == labels).sum().item()
                total += labels.size(0)

        acc = correct / total * 100
        scheduler.step()

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {train_loss/len(train_loader):.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "bgnet_model.pth")
            print(f"  → Saved best model (acc={acc:.2f}%)")

    print(f"\nTraining complete! Best Accuracy: {best_acc:.2f}%")
    print("Model saved as bgnet_model.pth")


if __name__ == "__main__":
    train()
