import os
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from utils import ImprovedThermalNet


class ThermalDataset(Dataset):
    def __init__(self, image_paths, labels, size=224):
        self.image_paths = image_paths
        self.labels = labels
        self.size = size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.random.rand(self.size, self.size) * 255
            img = img.astype(np.uint8)
        img = cv2.resize(img, (self.size, self.size))
        img = img.astype(np.float32) / 255.0
        img = torch.tensor(img).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, label


def train_model(model, train_loader, val_loader, epochs=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    print(f"\n🔧 Устройство: {device}")
    print("=" * 60)
    print("НАЧАЛО ОБУЧЕНИЯ")
    print("=" * 60)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_acc = 100 * correct / total
        scheduler.step()
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'models/best_model.pth')
        if (epoch + 1) % 10 == 0:
            print(f"Эпоха {epoch+1:3d}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}% | Best: {best_acc:.2f}%")
    print(f"\n✅ Обучение завершено! Лучшая точность: {best_acc:.2f}%")
    return model


def main():
    print("=" * 60)
    print("ОБУЧЕНИЕ НА ТЕРМОГРАММАХ")
    print("=" * 60)
    os.makedirs("models", exist_ok=True)
    labels_file = "real_labels.json"
    if not os.path.exists(labels_file):
        print(f"\n❌ Файл {labels_file} не найден!")
        print("Сначала запустите: python src/generate_data.py")
        return
    with open(labels_file, "r") as f:
        labels_dict = json.load(f)
    image_paths = list(labels_dict.keys())
    labels = list(labels_dict.values())
    print(f"\n📊 Загружено {len(image_paths)} изображений")
    X_train, X_val, y_train, y_val = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    train_dataset = ThermalDataset(X_train, y_train)
    val_dataset = ThermalDataset(X_val, y_val)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    model = ImprovedThermalNet(num_classes=3)
    model = train_model(model, train_loader, val_loader, epochs=100)
    print("\n💾 Модель сохранена: models/best_model.pth")


if __name__ == "__main__":
    main()
