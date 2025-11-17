# =====================================================
# MRI Tumor Classifier Training Script (GPU-Optimized)
# =====================================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

# =====================================================
# Safe Entry Point for Windows
# =====================================================
if __name__ == "__main__":
    # ======================================
    # Config
    # ======================================
    data_root = r"F:\MRI\Backend\Dataset"   # <-- parent folder containing 'tumor' and 'no_tumor'
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device} | CUDA available: {torch.cuda.is_available()}")

    # Enable fast GPU algorithms (auto-tuning)
    torch.backends.cudnn.benchmark = True

    # ======================================
    # Image Transform
    # ======================================
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])

    # ======================================
    # Dataset Loading
    # ======================================
    # Expecting structure:
    # Dataset/
    #   ├── no_tumor/
    #   └── tumor/
    train_dataset = datasets.ImageFolder(root=data_root, transform=transform)
    dataloader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,      # safe for Windows with this setup
        pin_memory=True
    )

    # ======================================
    # Model Setup
    # ======================================
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
    model = model.to(device)

    # For multi-GPU support (optional)
    if torch.cuda.device_count() > 1:
        print(f"[INFO] Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)

    # ======================================
    # Training Setup
    # ======================================
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002)
    epochs = 20

    # ======================================
    # Training Loop
    # ======================================
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        loop = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch [{epoch+1}/{epochs}]")

        for i, (imgs, labels) in loop:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    # ======================================
    # Save Trained Model
    # ======================================
    torch.save(model.state_dict(), "tumor_classifier.pth")
    print("✅ Training complete. Model saved as tumor_classifier.pth")
