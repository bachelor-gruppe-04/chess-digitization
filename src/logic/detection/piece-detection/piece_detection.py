import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import onnxruntime as ort
import onnx
from onnxsim import simplify

# Load ONNX model
model_path = "src/logic/models/480M_leyolo_pieces.onnx"
model = onnx.load(model_path)

# Extract class names from model metadata
metadata = model.metadata_props
class_names = {}

for item in metadata:
    if item.key == "names":
        # Parse the names field (assuming it's a JSON-like string)
        class_names = eval(item.value)  # Be cautious with eval(), consider using json.loads(item.value)
        break

# Print to check class names
print(class_names)

# For simplification of the model
model_simplified, check = simplify(model)

# Save the simplified model
simplified_model_path = "src/logic/models/480M_leyolo_pieces_simplified.onnx"
onnx.save(model_simplified, simplified_model_path)

# Now you can use the simplified model instead of the original
ort_session = ort.InferenceSession(simplified_model_path)

# Define image preprocessing function
def preprocess_image(image, target_width, target_height):
    """Resize and normalize the image for ONNX model inference."""
    image = cv2.resize(image, (target_width, target_height))  # Resize image
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.transpose(2, 0, 1)  # Convert HWC to CHW format
    return image[np.newaxis, ...].astype(np.float16)  # Add batch dimension and convert to float16

def predict(image, target_width=480, target_height=288, confidence_threshold=0.2):
    """Predict bounding boxes, class indices, and scores using the ONNX model."""
    # Preprocess the image
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
    predsT = np.transpose(preds, (0, 2, 1))  # Transpose to [1, 16, 2835]

    # Extract bounding box coordinates
    xc = predsT[:, :, 0]
    yc = predsT[:, :, 1]
    w = predsT[:, :, 2]
    h = predsT[:, :, 3]

    # Extract class probabilities (objectness is folded into the class scores)
    class_probs = predsT[:, :, 4:]  # Class probabilities (from index 4 onwards)

    # Extract class indices (max probability class)
    class_indices = np.argmax(class_probs, axis=-1)

    # Extract scores (using the class probabilities as the "score")
    scores = np.max(class_probs, axis=-1)

    # Apply confidence threshold
    mask = scores > confidence_threshold  # Create a mask for predictions above the threshold
    xc = xc[mask]
    yc = yc[mask]
    w = w[mask]
    h = h[mask]
    scores = scores[mask]
    class_indices = class_indices[mask]

    return xc, yc, w, h, scores, class_indices


def scale_boxes(xc, yc, w, h, orig_width, orig_height, target_width, target_height):
    """Scale coordinates back to the original image size."""
    xc = xc * (orig_width / target_width)
    yc = yc * (orig_height / target_height)
    w = w * (orig_width / target_width)
    h = h * (orig_height / target_height)
    return xc, yc, w, h

def apply_nms(boxes, scores, class_indices, nms_threshold=0.5):
    """Apply Non-Maximum Suppression to remove overlapping boxes."""
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=nms_threshold)
    indices = indices.flatten() if indices is not None else []
    boxes = boxes[indices]
    scores = scores[indices]
    class_indices = class_indices[indices]  # Apply mask to class indices as well
    return boxes, scores, class_indices


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

# Process video
video_path = 'resources/videos/chessvideo.mp4'
cap = cv2.VideoCapture(video_path)

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
print(frame_width)
print(frame_height)

# Create VideoWriter to save the processed video
output_path = 'resources/videos/output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for video
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process every second frame
    if frame_counter % 5 == 0:
        # Get predictions for the current frame
        xc, yc, w, h, scores, class_indices = predict(frame)

        # Scale bounding boxes back to original frame size
        xc, yc, w, h = scale_boxes(xc, yc, w, h, frame_width, frame_height, 480, 288)

        # Apply NMS to filter overlapping boxes
        boxes = np.column_stack((xc, yc, w, h))  # Combine xc, yc, w, h into boxes
        boxes, scores, class_indices = apply_nms(boxes, scores, class_indices)


        # Unpack the boxes back into separate variables after NMS
        xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

        # Visualize bounding boxes on the frame
        visualized_frame = visualize_boxes_and_labels(frame.copy(), xc, yc, w, h, class_indices, scores, class_names)

        # Write the frame to the output video
        out.write(visualized_frame)

        resized_frame = cv2.resize(visualized_frame, (1280, 720))
        cv2.imshow('Video', resized_frame)

    frame_counter += 1

    # Break the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
