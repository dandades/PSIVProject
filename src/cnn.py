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
    # Dataset
    data_dir = "dataset/team_classifier"
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Modelo
    model = TeamCNN().to(device)

    # Loss y optimizador
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Entrenamiento
    epochs = 10

    for epoch in range(epochs):
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

    # Guardar modelo
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/team_cnn.pth")

    print("Model saved at models/team_cnn.pth")


# Creamos un main para evitar que se entrene al importar
if __name__ == "__main__":
    train()