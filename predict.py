import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load architecture
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, 2)

# Load trained weights
state_dict = torch.load(r"D:\MRI\Backend\tumor_classifier.pth", map_location=device)
model.load_state_dict(state_dict)
model.to(device)

# Preprocessing + Prediction
def predict_mri(image_path: str):
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        conf, preds = torch.max(probs, 1)
        label = "Tumor Detected" if preds.item() == 1 else "No Tumor"

    return {"label": label, "confidence": float(conf.item())}
