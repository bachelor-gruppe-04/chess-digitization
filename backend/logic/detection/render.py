import cv2
import numpy as np
import tensorflow as tf

from typing import List, Tuple
from constants import MODEL_WIDTH, MODEL_HEIGHT



# def draw_box(frame: np.ndarray, color: Tuple[int, int, int], x: float, y: float, text: str, font_height: int) -> np.ndarray:
#     """
#     Draw a labeled box on an image.
#     """
#     x, y = int(x), int(y)
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     cv2.rectangle(frame, (x - font_height // 2, y - font_height // 2), (x + font_height // 2, y + font_height // 2), color, -1)  # Filled box
#     cv2.putText(frame, text, (x + 5, y - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White text

#     return frame

def draw_boxes_with_scores(
    frame: np.ndarray, 
    boxes, 
    scores,  
    threshold=0.5
):
    """
    Draws scaled bounding boxes and their max scores on the frame.
    Args:
        boxes: (N, 4) — assumed to be normalized to model resolution.
        scores: (N, num_classes)
        model_width: Input width of the model (e.g., 640)
        model_height: Input height of the model (e.g., 640)
    """
    boxes_np = boxes.numpy() if isinstance(boxes, tf.Tensor) else boxes
    scores_np = scores.numpy() if isinstance(scores, tf.Tensor) else scores

    frame_height, frame_width = frame.shape[:2]
    scale_x = frame_width / MODEL_WIDTH
    scale_y = frame_height / MODEL_HEIGHT

    for box, score_arr in zip(boxes_np, scores_np):
        max_score = np.max(score_arr)
        class_idx = np.argmax(score_arr)

        if max_score >= threshold:
            # Scale box coordinates
            l, t, r, b = box
            scaled_box = (
                l * scale_x,
                t * scale_y,
                r * scale_x,
                b * scale_y
            )

            color = (0, 100, 200)
            draw_box(frame, color, scaled_box, max_score)


def draw_box(frame: np.ndarray, color: Tuple[int, int, int], box: Tuple[float, float, float, float], score: float, font_height: int = 16) -> np.ndarray:
    """
    Draw a labeled box with score on the image.
    Args:
        frame: The image frame to draw on.
        color: The color of the box (B, G, R).
        box: Tuple of (left, top, right, bottom).
        score: Confidence score to display.
        font_height: Font size (affects box height).
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    l, t, r, b = map(int, box)

    # Draw filled rectangle for the box
    cv2.rectangle(frame, (l, t), (r, b), color, 2)

    # Prepare the score text
    text = f"{score:.2f}"

    # Measure text size
    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_w, text_h = text_size

    # Draw filled rectangle for text background
    cv2.rectangle(frame, (l, t - text_h - 4), (l + text_w + 4, t), color, -1)

    # Draw text on top of the box
    cv2.putText(frame, text, (l + 2, t - 2), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return frame




def draw_points(frame: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Draws the centers as circles on an OpenCV image (canvas) and returns the modified frame.
    
    :param canvas: The image or frame to draw on.
    :param centers: List of (x, y) coordinates, shape (64, 2).
    :return: The modified frame with drawn centers.
    """
    
    frame_height, frame_width = frame.shape[:2]
    
    for i, (x, y) in enumerate(points):  # Assuming shape (64, 2)
        x = round(x * frame_width / MODEL_WIDTH) 
        y = round(y * frame_height / MODEL_HEIGHT) 

        cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    return frame



def draw_polygon(frame, polygon):
    frame_height, frame_width = frame.shape[:2]
    
    sx = frame_width / MODEL_WIDTH
    sy = frame_height / MODEL_HEIGHT
    
    # Scale the polygon coordinates
    scaled_polygon = np.array([
        [(x * sx, y * sy) for x, y in polygon]
    ], dtype=np.int32)

    cv2.polylines(frame, scaled_polygon, isClosed=True, color=(0, 0, 255), thickness=2)

    return frame


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