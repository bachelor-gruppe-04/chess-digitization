import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import onnxruntime as ort

# Define class names
class_names = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 4: 'black-queen',
    5: 'black-rook', 6: 'white-bishop', 7: 'white-king', 8: 'white-knight', 9: 'white-pawn',
    10: 'white-queen', 11: 'white-rook'
}

# Load ONNX model (replace with your actual path)
model_path = "sandbox/piece-detection/models/480M_leyolo_pieces.onnx"
ort_session = ort.InferenceSession(model_path)

# Define image preprocessing function
def preprocess_image(image, target_width, target_height):
    """Resize and normalize the image for ONNX model inference."""
    image = cv2.resize(image, (target_width, target_height))  # Resize image
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.transpose(2, 0, 1)  # Convert HWC to CHW format
    return image[np.newaxis, ...].astype(np.float16)  # Add batch dimension and convert to float16

def predict(image_path, target_width=480, target_height=288, confidence_threshold=0.5):
    """Predict bounding boxes, class indices, and scores using the ONNX model."""
    # Load and preprocess the image
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image, target_width, target_height)

    # Get model inputs and outputs
    model_inputs = ort_session.get_inputs()
    model_outputs = ort_session.get_outputs()

    # Run inference
    predictions = ort_session.run(
        output_names=[output.name for output in model_outputs],
        input_feed={model_inputs[0].name: preprocessed_image}
    )

    # Extract predictions (assuming output shape [1, 2835, 16])
    preds = predictions[0]  # Get the first output (assuming it's the prediction output)
    predsT = np.transpose(preds, (0, 2, 1))  # Transpose to [1, 2835, 16]

    # Extract bounding box coordinates
    xc = predsT[:, :, 0]
    yc = predsT[:, :, 1]
    w = predsT[:, :, 2]
    h = predsT[:, :, 3]

    # Extract scores (objectness) and class probabilities
    scores = predsT[:, :, 4]  # Objectness score
    class_probs = predsT[:, :, 5:16]  # Class probabilities (assuming 12 classes)

    # Extract class indices (max probability class)
    class_indices = np.argmax(class_probs, axis=-1)

    # Apply confidence threshold
    mask = scores > confidence_threshold  # Create a mask for predictions above the threshold
    xc = xc[mask]
    yc = yc[mask]
    w = w[mask]
    h = h[mask]
    scores = scores[mask]
    class_indices = class_indices[mask]

    return xc, yc, w, h, scores, class_indices

def apply_nms(boxes, scores, nms_threshold=0.4):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=nms_threshold)
    indices = indices.flatten() if indices is not None else []
    boxes = boxes[indices]
    scores = scores[indices]
    return boxes, scores

def visualize_boxes_and_labels(image, xc, yc, w, h, class_indices, scores, class_names):
    """Visualizes bounding boxes and labels on an image."""
    for i in range(xc.shape[0]):  # Iterate over the predictions after NMS
        x_min = xc[i] - w[i] / 2
        y_min = yc[i] - h[i] / 2
        x_max = xc[i] + w[i] / 2
        y_max = yc[i] + h[i] / 2

        # Draw bounding box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        # Get class label and score
        class_name = class_names[class_indices[i]]
        score = scores[i]

        # Add label to the bounding box
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image, label, (int(x_min), int(y_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 50, 50), 2)

    return image

# Load the image
image_path = 'sandbox/piece-detection/images/chessboard.jpg'  # Replace with your actual image path

# Get predictions
xc, yc, w, h, scores, class_indices = predict(image_path)

# Apply NMS to filter overlapping boxes
boxes = np.column_stack((xc, yc, w, h))  # Combine xc, yc, w, h into boxes
boxes, scores = apply_nms(boxes, scores, nms_threshold=0.4)

# Unpack the boxes back into separate variables after NMS
xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

# Load the original image for visualization (in BGR format)
image = cv2.imread(image_path)

# Visualize the boxes and labels
visualized_image = visualize_boxes_and_labels(image.copy(), xc, yc, w, h, class_indices, scores, class_names)

# Convert BGR to RGB for Matplotlib
visualized_image = cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB)

# Display the image using Matplotlib
plt.figure(figsize=(10, 6))
plt.imshow(visualized_image)
plt.axis('off')
plt.show()
