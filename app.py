import gradio as gr
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from modelo import ReconhecedorFacialCNN

# 1. Model & Labels Setup
# Ensure these match the alphabetical order of your training folders
labels = [
    'Albert Einstein', 'Angelina Jolie', 'Barack Obama', 'Bolsonaro', 
    'Elon Musk', 'Linus Torvalds', 'Lula', 'Marie Curie', 
    'Nikola Tesla', 'Taylor Swift', 'Trump', 'Will Smith', 'Zendaya'
]

# Initialize architecture
model = ReconhecedorFacialCNN(num_classes=len(labels))

# Load weights (map_location='cpu' is required for the free tier)
model.load_state_dict(torch.load("modelo_cientistas.pth", map_location=torch.device('cpu')))
model.eval()

# 2. Prediction Function
def predict(img):
    if img is None:
        return None
    
    # Preprocessing pipeline
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    img_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        probabilities = F.softmax(output[0], dim=0)
    
    # Return a dictionary of probabilities for the Gradio Label component
    return {labels[i]: float(probabilities[i]) for i in range(len(labels))}

# 3. Gradio Interface Construction

interface = gr.Interface(
    fn=predict, 
    inputs=gr.Image(type="pil", label="Upload Photo"), 
    outputs=gr.Label(num_top_classes=3, label="Predictions"),
    title="Personality AI Detector v1.0",
    description="An AI trained to identify famous scientists, politicians, and celebrities using a Custom CNN.",
    article="Developed as a deep learning computer vision project. Classes include Einstein, Musk, Tesla, and more.",
)

if __name__ == "__main__":
    interface.launch()
