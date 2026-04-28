import torch
import torch.nn as nn                           # neural network layers
import torch.optim as optim                     # optimizers (how the network learns)
from torchvision import datasets, transforms    # image tools
from torch.utils.data import DataLoader         # batch data loading

import os

transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()])

# Dataset
data_dir = "dataset/team_classifier"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Estamos creando a un jugador con IA (bot de Fortnite)
class TeamCNN(nn.Module):
    def __init__(self):
        super(TeamCNN, self).__init__()

        # Los OJOS del jugador (sistema de visión)
        self.conv = nn.Sequential(
            # Primera capa
            nn.Conv2d(3, 16, 3, padding=1), # El bot mira cosas simples: colores, bordes, formas --> "veo algo rojo", "veo una silueta"
            nn.ReLU(),                      # lo importante lo guardo, lo raro lo ignoro
            nn.MaxPool2d(2),                # no quiero ver cada píxel, solo la idea general

            # Segunda capa (jugador más listo)
            nn.Conv2d(16, 32, 3, padding=1), # El bot dice: "esto parece un jugador completo"
            nn.ReLU(), # filtra +
            nn.MaxPool2d(2), # simplifica +

            nn.Conv2d(32, 64, 3, padding=1), # El bot ya reconoce la skin del jugador y su forma de jugar (es un tryhard o un noob?)
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Aquí el bot ya piensa y decide
        self.fc = nn.Sequential(
            nn.Flatten(), # Reduce todo lo que sabe una lista de pro tips de Keroro
            nn.Linear(64 * 16 * 16, 128), # Lo reduce todo a solo 128 pro tips
            nn.ReLU(), # Se queda solo con los pro tips útiles (mecánicas, aim y gamesense)
            nn.Linear(128, 2)  # O es del equipo azul o del rojo (red vs blue)
        )

    # ¿Qué hace el jugador en una partida? Mira la pantalla, analiza detalles, decide a qué equipo pertenece el jugador, responde (si es aliado le doy un medkit, si es rival le disparo)
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Si tengo un PC tocho (GPU), jugamos en gráficos ultra

model = TeamCNN().to(device) # creamos el bot y lo metemos en la máquina

criterion = nn.CrossEntropyLoss() # el coach me dice qué tan bien o mal he jugado
optimizer = optim.Adam(model.parameters(), lr=0.001) # si fallo me dice cómo mejorar y lo aplico

# Training loop
epochs = 10 # Jugamos 10 partidas

for epoch in range(epochs): # En cada partida
    total_loss = 0

    for images, labels in train_loader: # Le enseño 32 jugadas al bot
        images, labels = images.to(device), labels.to(device)

        outputs = model(images) # Cree que es aliado o enemigo
        loss = criterion(outputs, labels) # ¿He acertado? | loss grande: He jugado muy mal -> Medio: Ahora he mejorado -> Bajo: Finalmente soy bastante bueno

        optimizer.zero_grad()
        loss.backward() # Me entero de que he fallado
        optimizer.step() # La próxima vez lo tendré en cuenta para jugar mejor

        total_loss += loss.item() 

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# Save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/team_cnn.pth")

print("Model saved at models/team_cnn.pth")