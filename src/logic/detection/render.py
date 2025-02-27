import cv2
import matplotlib.pyplot as plt
import numpy as np
from constants import MODEL_WIDTH, MODEL_HEIGHT

def draw_points(frame, centers, color, sx, sy):
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
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blå farge i BGR-format

    return frame


def draw_polygon(frame, boundary, color, sx, sy):
    """
    Draw a polygon on an image using OpenCV.
    """
    scaled_boundary = np.array([(int(x * sx), int(y * sy)) for x, y in boundary], np.int32)
    cv2.polylines(frame, [scaled_boundary], isClosed=True, color=color, thickness=2)

def draw_box(frame, color, x, y, text, font_height, line_width):
    """
    Draw a labeled box on an image.
    """
    x, y = int(x), int(y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (x - font_height // 2, y - font_height // 2), (x + font_height // 2, y + font_height // 2), color, -1)  # Filled box
    cv2.putText(frame, text, (x + 5, y - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White text



def visualize_centers_opencv(canvas, centers):
    """
    Draws the centers as circles on an OpenCV image (canvas) and returns the modified frame.
    Scales the coordinates according to the given model dimensions.
    
    :param canvas: The image or frame to draw on.
    :param centers: List of (x, y) coordinates.
    :param model_width: Width of the model (for scaling).
    :param model_height: Height of the model (for scaling).
    :return: The modified frame with drawn centers.
    """

    canvas_height, canvas_width, _ = canvas.shape
    
    for center in centers:
        x = int(center[0] * canvas_width/MODEL_WIDTH)
        y = int(center[1] * canvas_height/MODEL_HEIGHT)

        cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)  # Blå farge i BGR-format

    
    return canvas  # Return the modified image