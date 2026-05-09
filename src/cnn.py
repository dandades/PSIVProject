import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from team_model import TeamCNN
import os

def train():
    # Carpeta donde auto_label.py ha organizado las imágenes (0, 1)
    data_dir = "dataset/team_classifier_auto"
    
    # Verificación de que el dataset existe
    if not os.path.exists(data_dir):
        print(f"Error: No se encuentra la carpeta {data_dir}. Ejecuta primero auto_label.py")
        return

    # TRANSFORMACIONES: Aumentamos la robustez para que la CNN generalice mejor
    # Usamos Normalización estándar de ImageNet que ayuda a la convergencia del modelo
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Robustez lumínica
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Carga del dataset (ignora automáticamente la carpeta 'duda')
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # Si auto_label fue muy estricto, avisamos si hay pocos datos
    if len(dataset) < 10:
        print(f"Advertencia: El dataset tiene solo {len(dataset)} imágenes. Podría haber overfitting.")

    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Configuración de dispositivo (GPU si está disponible)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Entrenando en: {device}")

    # Inicialización del modelo, criterio de pérdida y optimizador
    model = TeamCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    # Usamos un Learning Rate conservador para no "destruir" los pesos con el ruido residual
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 

    epochs = 15
    print("Iniciando entrenamiento...")
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}")

    # GUARDADO: Coherencia absoluta con track_and_classify.py
    os.makedirs("models", exist_ok=True)
    model_path = "models/team_cnn.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"Entrenamiento finalizado. Modelo guardado en: {model_path}")

if __name__ == "__main__":
    train()