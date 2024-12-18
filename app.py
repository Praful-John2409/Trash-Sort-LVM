import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPModel
from PIL import Image
import gradio as gr

# Define class labels
class_labels = ["Trash", "Compostable", "Recyclable"]

# Define CLIP Classifier (same as used during training)
class CLIPClassifier(nn.Module):
    def __init__(self, clip_model, num_classes):
        super(CLIPClassifier, self).__init__()
        self.clip = clip_model.vision_model
        self.fc = nn.Linear(768, num_classes)

    def forward(self, images):
        image_features = self.clip(images).pooler_output
        return self.fc(image_features)

# Load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPClassifier(clip_model, num_classes=3).to(device)

# Load the saved weights
model.load_state_dict(torch.load("clip_trash_classifier_finetuned.pth", map_location=device))
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],  # CLIP's mean
                         std=[0.26862954, 0.26130258, 0.27577711])  # CLIP's std
])

# Prediction function
def predict(image):
    """
    Function to predict the class label and confidence of the uploaded image.
    Returns separate values for label and confidence.
    """
    # Preprocess the image
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        confidence, predicted = torch.max(probabilities, dim=0)

    # Get predicted class and confidence score
    predicted_class = class_labels[predicted.item()]
    confidence_score = f"{confidence.item() * 100:.2f}%"

    # Return as separate outputs
    return predicted_class, confidence_score

# Gradio Interface
interface = gr.Interface(
    fn=predict,                    # Prediction function
    inputs=gr.Image(type="pil"),   # Input: Image in PIL format
    outputs=[
        gr.Textbox(label="Predicted Category"),  # Output 1: Predicted Label
        gr.Textbox(label="Confidence")           # Output 2: Confidence Score
    ],
    title="Trash Classifier Using CLIP",
    description="Upload an image to classify it as **Trash**, **Compostable**, or **Recyclable**.\n"
                "The app will display the predicted category and confidence score."
)

# Launch the app
if __name__ == "__main__":
    interface.launch(share=True)