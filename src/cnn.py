import torch
import torch.nn as nn                           # neural network layers
import torch.optim as optim                     # optimizers (how the network learns)
from torchvision import datasets, transforms    # image tools
from torch.utils.data import DataLoader         # batch data loading

import os

# Transformations (before training the network, images must be the same size)
transform = transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor()]) # ToTensor converts an image to a tensor (numeric matrix) | RGB pixel -> [0.2, 0.8, 0.1]

# Dataset
data_dir = "dataset/team_classifier"
dataset = datasets.ImageFolder(root=data_dir, transform=transform)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True) # batches of 32 images (shuffled)

# Simple CNN (THE MODEL)
class TeamCNN(nn.Module):
    def __init__(self):
        super(TeamCNN, self).__init__()

        # CONVOLUTIONAL LAYERS
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), # 3 RGB channels, 16 filters that learn, 3 kernel (window 3x3), padding=1 --> maintains size
            nn.ReLU(),                      # introduces nonlinearity
            nn.MaxPool2d(2),                # reduces the size to its half

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # FULLY CONNECTED (Here, the network decides if it's A or B)
        self.fc = nn.Sequential(
            nn.Flatten(), # Trnsforms multidimensional data into a one-dimensional vector
            nn.Linear(64 * 16 * 16, 128), # reduces information to 128 neurons
            nn.ReLU(),
            nn.Linear(128, 2)  # A or B
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

# Training setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TeamCNN().to(device) # sends the model to the CPU or GPU

criterion = nn.CrossEntropyLoss() # measures the accuracy of the prediction
optimizer = optim.Adam(model.parameters(), lr=0.001) # updates the model

# Training loop
epochs = 10 # 10 complete loops of the dataset

for epoch in range(epochs):
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device) # move to GPU

        outputs = model(images) # network guesses A or B
        loss = criterion(outputs, labels) # how wrong the network was?

        optimizer.zero_grad() # reset gradients
        loss.backward() # backpropagation
        optimizer.step() # update weigths

        total_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f}")

# Save the model
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/team_cnn.pth")

print("Model saved at models/team_cnn.pth")