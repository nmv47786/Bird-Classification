import grpc
import numpy as np
import cv2
import tritonclient.grpc as grpcclient
import json

def load_label_mapping():
    with open("triton/model_repository/bird_classifier/config.json", 'r') as file:
        data = json.load(file)
        return data['id2label']

def preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = np.asarray(image, dtype=np.float32) / 255.0
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0)
    return image

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

def perform_inference(input_data):
    triton_client = grpcclient.InferenceServerClient(url="localhost:8001", verbose=False)
    input_name = "onnx::Pad_0"
    output_name = "1047"
    inputs = [grpcclient.InferInput(input_name, input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [grpcclient.InferRequestedOutput(output_name)]
    response = triton_client.infer("bird_classifier", inputs=inputs, outputs=outputs)
    output_data = response.as_numpy(output_name)
    if output_data.dtype != np.float32:
        raise ValueError("Expected model output of type float32, received:", output_data.dtype)
    probabilities = softmax(output_data)
    return probabilities

def get_top_predictions(probabilities, labels, top_k=3):
    top_indices = np.argsort(probabilities[0])[-top_k:][::-1]
    results = [(labels[str(i)], probabilities[0][i]) for i in top_indices]
    return results

if __name__ == "__main__":
    labels = load_label_mapping()
    image_path = "assets/barnowl.jpg"
    input_data = preprocess_image(image_path)
    probabilities = perform_inference(input_data)
    top_predictions = get_top_predictions(probabilities, labels)
    for species, probability in top_predictions:
        print(f"Predicted Bird Species: {species}, Probability: {probability:.4f}")
