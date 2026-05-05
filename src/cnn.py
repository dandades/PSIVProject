import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os

from team_model import TeamCNN

# Transformaciones (mismo tamaño que usaremos en inferencia)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])


def train():
    # dataset directory
    data_dir = "dataset/team_classifier_auto"

    dataset = datasets.ImageFolder(
        root=data_dir,
        transform=transform,
        is_valid_file=lambda x: x.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        )
    )

    # Mensajes informativos
    print("Classes:", dataset.classes)
    print("Class mapping:", dataset.class_to_idx)
    print("Total images:", len(dataset))

    if len(dataset) == 0:
        raise ValueError(
            "Dataset vacío. Revisa dataset/team_classifier_auto (0 y 1 con imágenes dentro)"
        )

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # MODELO
    model = TeamCNN().to(device)

    # LOSS + OPTIMIZER
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento con 10 epochs
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

    # Guardamos el modelo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/team_cnn.pth")

    print("Model saved at models/team_cnn.pth")


if __name__ == "__main__":
    train()