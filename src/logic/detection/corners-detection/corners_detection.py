import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import onnxruntime as ort
import onnx
from onnxsim import simplify

# Load ONNX models for both object detection and corner detection
model_path = "sandbox/models/480M_leyolo_pieces.onnx"
corner_model_path = "sandbox/models/480L_leyolo_xcorners.onnx"

# Load ONNX sessions
ort_session = ort.InferenceSession(model_path)
corner_ort_session = ort.InferenceSession(corner_model_path)

model = onnx.load(model_path)

metadata = model.metadata_props
class_names = {}

for item in metadata:
    if item.key == "names":
        # Parse the names field (assuming it's a JSON-like string)
        class_names = eval(item.value)  # Be cautious with eval(), consider using json.loads(item.value)
        break

# Print to check class names
print(class_names)

# Define the image preprocessing functions for both models
def preprocess_image(image, target_width, target_height):
    """Resize and normalize the image for ONNX model inference."""
    image = cv2.resize(image, (target_width, target_height))  # Resize image
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.transpose(2, 0, 1)  # Convert HWC to CHW format
    return image[np.newaxis, ...].astype(np.float16)  # Add batch dimension and convert to float16

def preprocess_corner_image(image, target_width, target_height):
    """Resize and normalize the image for corner detection ONNX model."""
    image = cv2.resize(image, (target_width, target_height))  # Resize image
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.transpose(2, 0, 1)  # Convert HWC to CHW format
    return image[np.newaxis, ...].astype(np.float16)  # Add batch dimension and convert to float16

# Define prediction functions for both models
def predict(image, target_width=480, target_height=288, confidence_threshold=0.2):
    """Predict bounding boxes, class indices, and scores using the ONNX model."""
    # Preprocess the image
    preprocessed_image = preprocess_image(image, target_width, target_height)

    # Run inference for object detection
    model_inputs = ort_session.get_inputs()
    model_outputs = ort_session.get_outputs()
    predictions = ort_session.run(
        output_names=[output.name for output in model_outputs],
        input_feed={model_inputs[0].name: preprocessed_image}
    )

    preds = predictions[0]  # Extract predictions
    predsT = np.transpose(preds, (0, 2, 1))  # Transpose to [1, 16, 2835]

    # Extract bounding box coordinates and class scores
    xc = predsT[:, :, 0]
    yc = predsT[:, :, 1]
    w = predsT[:, :, 2]
    h = predsT[:, :, 3]
    class_probs = predsT[:, :, 4:]
    class_indices = np.argmax(class_probs, axis=-1)
    scores = np.max(class_probs, axis=-1)

    # Apply confidence threshold
    mask = scores > confidence_threshold
    xc = xc[mask]
    yc = yc[mask]
    w = w[mask]
    h = h[mask]
    scores = scores[mask]
    class_indices = class_indices[mask]

    return xc, yc, w, h, scores, class_indices

def predict_corners(image, target_width=480, target_height=288, confidence_threshold=0.2):
    """Predict corner points using the ONNX corner detection model."""
    preprocessed_image = preprocess_corner_image(image, target_width, target_height)

    # Run inference for corner detection
    model_inputs = corner_ort_session.get_inputs()
    model_outputs = corner_ort_session.get_outputs()
    predictions = corner_ort_session.run(
        output_names=[output.name for output in model_outputs],
        input_feed={model_inputs[0].name: preprocessed_image}
    )

    corners = predictions[0]  # Adjust based on your model's output format
    return corners

# Function to visualize bounding boxes
def visualize_boxes(image, xc, yc, w, h, class_indices, scores, class_names):
    """Visualizes bounding boxes and labels on an image."""
    for i in range(xc.shape[0]):
        x_min = xc[i] - w[i] / 2
        y_min = yc[i] - h[i] / 2
        x_max = xc[i] + w[i] / 2
        y_max = yc[i] + h[i] / 2
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        class_name = class_names[class_indices[i]]
        score = scores[i]
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image, label, (int(x_min), int(y_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 50, 50), 2)
    return image

# Function to visualize corners
def visualize_corners(image, corners, target_width=480, target_height=288, confidence_threshold=0.2):
    """Visualize corner points on the image."""
    frame_height, frame_width = image.shape[:2]
    for corner in corners.T:  # Assuming corners are transposed correctly
        x, y, w, h, conf = corner

        if conf < confidence_threshold:
            continue  # Skip low-confidence predictions

        x = int(x * frame_width / target_width)
        y = int(y * frame_height / target_height)

        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    return image

# Process video and integrate both object detection and corner detection
video_path = 'sandbox/videos/chessvideo.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = 'sandbox/videos/output_video_combined.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % 5 == 0:
        # Object detection predictions
        xc, yc, w, h, scores, class_indices = predict(frame)

        # Corner detection predictions
        corners = predict_corners(frame)

        # Visualize both object boxes and corners
        visualized_frame = visualize_boxes(frame.copy(), xc, yc, w, h, class_indices, scores, class_names)
        visualized_frame = visualize_corners(visualized_frame, corners)

        # Write the frame to the output video
        out.write(visualized_frame)

        resized_frame = cv2.resize(visualized_frame, (1280, 720))
        cv2.imshow('Video with Object Detection and Corners', resized_frame)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
