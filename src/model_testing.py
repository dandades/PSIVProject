import torch
from PIL import Image
from torchvision import transforms

model = TeamCNN()
model.load_state_dict(torch.load("models/team_cnn.pth"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

img = Image.open("test.jpg")
img = transform(img).unsqueeze(0)

output = model(img)
pred = torch.argmax(output)

print("Equipo:", pred.item())