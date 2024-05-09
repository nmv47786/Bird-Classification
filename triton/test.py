from transformers import AutoModelForImageClassification
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# Load the image classification model
model = AutoModelForImageClassification.from_pretrained("chriamue/bird-species-classifier")

def preprocess_image(image_path):
    # Load image
    image = cv2.imread(image_path)
    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Resize image to the expected input size of the model (224x224)
    image = cv2.resize(image, (224, 224))
    # Normalize pixel values to [0, 1], assuming model was trained this way
    image = image.astype(np.float32) / 255.0
    # Reorder dimensions to match what the model expects: (batch_size, channels, height, width)
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return torch.tensor(image)  # Convert to PyTorch tensor

# Preprocess the image
image_path = "assets/robin.jpg"
inputs = preprocess_image(image_path)

# Perform inference
with torch.no_grad():
    outputs = model(inputs)

# Apply softmax to convert logits to probabilities
probabilities = F.softmax(outputs.logits, dim=1)

# Get the predicted class index and confidence
predicted_idx = probabilities.argmax()
confidence = probabilities[0, predicted_idx]

# Print the predicted bird species and confidence
predicted_species = model.config.id2label[predicted_idx.item()]
print(f"Predicted Bird Species: {predicted_species}")
print(f"Confidence: {confidence.item():.4f}")
