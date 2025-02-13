import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import onnxruntime as ort
import onnx
from onnxsim import simplify


#**IMPORTANT:** Break the loop by pressing 'Q'!!

def preprocess_image(image, target_width, target_height):
    """Resize and normalize the image for ONNX model inference."""
    image = cv2.resize(image, (target_width, target_height))  
    image = image / 255.0 
    image = image.transpose(2, 0, 1)  
    return image[np.newaxis, ...].astype(np.float16) 

def predict_pieces(image, ort_session, target_width=480, target_height=288, confidence_threshold=0.2):
    """
    Predict bounding boxes, class indices, and scores using the ONNX model.

    Args:
        image (numpy array): Input image for prediction.
        ort_session: ONNX Runtime InferenceSession object.
        target_width (int): Width to resize the input image to before inference.
        target_height (int): Height to resize the input image to before inference.
        confidence_threshold (float): Minimum score to keep predictions.

    Returns:
        tuple: Bounding box coordinates (xc, yc, w, h), confidence scores, and class indices.
    """
    preprocessed_image = preprocess_image(image, target_width, target_height)

    model_inputs = ort_session.get_inputs()
    model_outputs = ort_session.get_outputs()

    # Run inference
    predictions = ort_session.run(
        output_names=[output.name for output in model_outputs],
        input_feed={model_inputs[0].name: preprocessed_image}
    )

    preds = predictions[0]
    predsT = np.transpose(preds, (0, 2, 1)) # Transpose to [1, 16, 2835]

    # Extract bounding box coordinates and other information
    xc = predsT[:, :, 0]
    yc = predsT[:, :, 1]
    w = predsT[:, :, 2]
    h = predsT[:, :, 3]

    # Extract class probabilities (objectness is folded into the class scores)
    class_probs = predsT[:, :, 4:]

    # Extract class indices (max probability class)
    class_indices = np.argmax(class_probs, axis=-1)

    # Extract scores (using the class probabilities as the "score")
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

def preprocess_image(image, target_width, target_height):
    """Resize and normalize the image for ONNX model inference."""
    image = cv2.resize(image, (target_width, target_height))
    image = image / 255.0 
    image = image.transpose(2, 0, 1)
    return image[np.newaxis, ...].astype(np.float16)

def scale_boxes(xc, yc, w, h, orig_width, orig_height, target_width, target_height):
    """
    Scale the bounding box coordinates back to the original image size.

    Args:
        xc, yc, w, h (numpy array): The bounding box coordinates.
        orig_width, orig_height (int): Original image dimensions.
        target_width, target_height (int): Target image dimensions used for resizing.

    Returns:
        tuple: Scaled bounding box coordinates.
    """
    xc = xc * (orig_width / target_width)
    yc = yc * (orig_height / target_height)
    w = w * (orig_width / target_width)
    h = h * (orig_height / target_height)
    return xc, yc, w, h

def apply_nms(boxes, scores, class_indices, nms_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.

    Args:
        boxes (numpy array): Bounding box coordinates.
        scores (numpy array): Confidence scores for each bounding box.
        class_indices (numpy array): Predicted class indices for each box.
        nms_threshold (float): Threshold to use for NMS.

    Returns:
        tuple: Filtered bounding boxes, scores, and class indices.
    """
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=nms_threshold)
    indices = indices.flatten() if indices is not None else []
    boxes = boxes[indices]
    scores = scores[indices]
    class_indices = class_indices[indices]
    return boxes, scores, class_indices


def visualize_boxes_and_labels(image, xc, yc, w, h, class_indices, scores, class_names):
    """
    Visualize bounding boxes and labels on the image.

    Args:
        image (numpy array): Image to draw the bounding boxes on.
        xc, yc, width, height (numpy array): Coordinates and size of the bounding boxes.
        class_indices (numpy array): Indices of the predicted classes.
        scores (numpy array): Confidence scores for each bounding box.
        class_names (dict): Dictionary mapping class indices to class names.

    Returns:
        numpy array: Image with bounding boxes drawn.
    """
    for i in range(xc.shape[0]):
        # Calculate the coordinates of the bounding box
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