import cv2
import numpy as np
import tensorflow as tf

from typing import List, Tuple
from utilities.constants import MODEL_WIDTH, MODEL_HEIGHT


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

    cv2.rectangle(frame, (l, t), (r, b), color, 2)

    text = f"{score:.2f}"

    text_size = cv2.getTextSize(text, font, 0.5, 1)[0]
    text_w, text_h = text_size

    cv2.rectangle(frame, (l, t - text_h - 4), (l + text_w + 4, t), color, -1)

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
    
    for i, (x, y) in enumerate(points):
        x = round(x * frame_width / MODEL_WIDTH) 
        y = round(y * frame_height / MODEL_HEIGHT) 

        cv2.circle(frame, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    return frame



def draw_polygon(frame, polygon):
    """
    Draws a scaled polygon on the given frame.

    The polygon coordinates are assumed to be normalized to the input model resolution
    (MODEL_WIDTH x MODEL_HEIGHT), and will be scaled to match the actual frame dimensions.

    Args:
        frame (np.ndarray): The image/frame on which to draw the polygon.
        polygon (List[Tuple[float, float]]): A list of (x, y) coordinates representing the polygon vertices.
                                             Coordinates should be normalized to the model input resolution.

    Returns:
        np.ndarray: The frame with the polygon drawn on it.
    """
    frame_height, frame_width = frame.shape[:2]
    
    sx = frame_width / MODEL_WIDTH
    sy = frame_height / MODEL_HEIGHT

    # Scale the polygon coordinates
    scaled_polygon = np.array([
        [(x * sx, y * sy) for x, y in polygon]
    ], dtype=np.int32)

    cv2.polylines(frame, scaled_polygon, isClosed=True, color=(0, 0, 255), thickness=2)

    return frame