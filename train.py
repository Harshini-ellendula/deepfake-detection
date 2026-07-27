import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Subset
import random
import os

def main():
    print("Training started...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # --- Data Transforms ---
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # --- Load datasets ---
    train_dataset = datasets.ImageFolder("dataset/train", transform=train_transform)
    val_dataset = datasets.ImageFolder("dataset/validation", transform=val_transform)

    print("Classes:", train_dataset.classes)
    print("Full training images:", len(train_dataset))

    subset_size = min(25000, len(train_dataset))
    indices = random.sample(range(len(train_dataset)), subset_size)
    train_dataset = Subset(train_dataset, indices)
    print("Using subset size:", len(train_dataset))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)  # <--- num_workers=0 for Windows
    val_loader = DataLoader(val_dataset, batch_size=32, num_workers=0)

    # --- Load model ---
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in model.parameters():
        param.requires_grad = False
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

    epochs = 10
    os.makedirs("models", exist_ok=True)

    # --- Training loop ---
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (images, labels) in enumerate(train_loader, 1):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if batch_idx % 100 == 0:
                print(f"[Epoch {epoch+1}/{epochs}] Batch {batch_idx}/{len(train_loader)} "
                      f"Loss: {running_loss/batch_idx:.4f} Accuracy: {100*correct/total:.2f}%")

        # --- Epoch complete ---
        accuracy = 100 * correct / total
        epoch_loss = running_loss / len(train_loader)
        print(f"=== Epoch [{epoch+1}/{epochs}] Complete: Loss={epoch_loss:.4f}, Accuracy={accuracy:.2f}% ===")

        # --- Save checkpoint ---
        checkpoint_path = f"models/deepfake_model_epoch{epoch+1}.pth"
        torch.save(model.state_dict(), checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Training complete. Final model saved.")
    torch.save(model.state_dict(), "models/deepfake_model_final.pth")


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Needed for Windows
    main()
