import torch
import cv2
from torchvision import transforms
from team_model import TeamCNN

# Transform igual que entrenamiento
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Cargar modelo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = TeamCNN().to(device)
model.load_state_dict(torch.load("models/team_cnn.pth", map_location=device))
model.eval()

def classify_team(image):
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        pred = torch.argmax(output, dim=1).item()

    return "Team A" if pred == 0 else "Team B"