from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import json
import tritonclient.grpc as grpcclient

app = Flask(__name__)
CORS(app)


def load_label_mapping():
    with open("triton/model_repository/bird_classifier/config.json", "r") as file:
        data = json.load(file)
        return data["id2label"]


def preprocess_image(image_file):
    in_memory_file = image_file.read()
    image_array = np.frombuffer(in_memory_file, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = (image - mean) / std
    image = np.transpose(image, (2, 0, 1))
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)


def perform_inference(input_data):
    triton_client = grpcclient.InferenceServerClient(url="triton:8001", verbose=False)
    input_name = "onnx::Pad_0"
    output_name = "1047"
    inputs = [grpcclient.InferInput(input_name, input_data.shape, "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [grpcclient.InferRequestedOutput(output_name)]
    response = triton_client.infer("bird_classifier", inputs=inputs, outputs=outputs)
    output_data = response.as_numpy(output_name)
    if output_data.dtype != np.float32:
        raise ValueError(
            "Expected model output of type float32, received:", output_data.dtype
        )
    probabilities = softmax(output_data)
    return probabilities


def get_top_predictions(probabilities, labels, top_k=3):
    top_indices = np.argsort(probabilities[0])[-top_k:][::-1]
    results = [(labels[str(i)], float(probabilities[0][i])) for i in top_indices]
    return results


@app.route("/api/identify", methods=["POST"])
def identify_bird():
    if "birdImage" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["birdImage"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400
    if file:
        labels = load_label_mapping()
        input_data = preprocess_image(file)
        probabilities = perform_inference(input_data)
        top_predictions = get_top_predictions(probabilities, labels)
        response_data = [
            {"species": label, "probability": prob} for label, prob in top_predictions
        ]
        return jsonify(response_data)
    else:
        return jsonify({"error": "Invalid input"}), 400


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
