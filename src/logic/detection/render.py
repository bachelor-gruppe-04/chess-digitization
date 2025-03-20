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



def visualize_centers_opencv(canvas, centers, confidence_threshold=0.2):
    """
    Draws the centers as circles on an OpenCV image (canvas) and returns the modified frame.
    Scales the coordinates according to the given model dimensions.
    
    :param canvas: The image or frame to draw on.
    :param centers: List of (x, y) coordinates.
    :param model_width: Width of the model (for scaling).
    :param model_height: Height of the model (for scaling).
    :return: The modified frame with drawn centers.
    """

    print("Shape of centers:", np.array(centers).shape)

    frame_height, frame_width = canvas.shape[:2]
    for corner in centers.T: 
        x, y, w, h, conf = corner

        if conf < confidence_threshold:
            continue 

        x = int(x * frame_width /MODEL_WIDTH)
        y = int(y * frame_height / MODEL_HEIGHT)

        cv2.circle(canvas, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    return canvas



# def visualize_corners(image, corners, target_width=480, target_height=288, confidence_threshold=0.2):
#     """
#     Visualizes corner points on an image.

#     Args:
#         image (numpy.ndarray): The image on which to visualize corners.
#         corners (numpy.ndarray): Predicted corner points (coordinates and confidence scores).
#         target_width (int, optional): The width to resize the image. Default is 480.
#         target_height (int, optional): The height to resize the image. Default is 288.
#         confidence_threshold (float, optional): The threshold for filtering low-confidence corner predictions. Default is 0.2.

#     Returns:
#         numpy.ndarray: The image with corner points visualized.
#     """
#     frame_height, frame_width = image.shape[:2]
#     for corner in corners.T: 
#         x, y, w, h, conf = corner

#         if conf < confidence_threshold:
#             continue 

#         x = int(x * frame_width / target_width)
#         y = int(y * frame_height / target_height)

#         cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
#     return image