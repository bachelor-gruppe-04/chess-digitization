import cv2
import numpy as np
from constants import MODEL_WIDTH, MODEL_HEIGHT

def draw_points(frame, centers, sx, sy):
    """
    Draw points on an image using OpenCV.
    
    - frame: Image to draw on
    - centers: List of (x, y) points
    - color: (B, G, R) color tuple
    - sx, sy: Scaling factors
    """
    for center in centers:
        x = int(center[0] * sx)
        y = int(center[1] * sy)
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1) 

    return frame


def draw_box(frame, color, x, y, text, font_height):
    """
    Draw a labeled box on an image.
    """
    x, y = int(x), int(y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (x - font_height // 2, y - font_height // 2), (x + font_height // 2, y + font_height // 2), color, -1)  # Filled box
    cv2.putText(frame, text, (x + 5, y - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White text


def visualize_centers(canvas, centers):
    """
    Draws the centers as circles on an OpenCV image (canvas) and returns the modified frame.
    
    :param canvas: The image or frame to draw on.
    :param centers: List of (x, y) coordinates, shape (64, 2).
    :param confidence_threshold: Threshold for drawing points.
    :return: The modified frame with drawn centers.
    """

    frame_height, frame_width = canvas.shape[:2]

    for i, (x, y) in enumerate(centers):  # Assuming shape (64, 2)

        x = round(x * frame_width / MODEL_WIDTH) 
        y = round(y * frame_height / MODEL_HEIGHT) 

        cv2.circle(canvas, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    return canvas
