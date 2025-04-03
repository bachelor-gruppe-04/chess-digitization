import cv2
import numpy as np
from typing import List, Tuple
from constants import MODEL_WIDTH, MODEL_HEIGHT, LABELS, PALETTE


# def draw_points(frame: np.ndarray, centers: List[Tuple[float, float]], sx: float, sy: float) -> np.ndarray:
#     """
#     Draw points on an image using OpenCV.
    
#     - frame: Image to draw on
#     - centers: List of (x, y) points
#     - sx, sy: Scaling factors
#     """
#     for center in centers:
#         x = int(center[0] * sx)
#         y = int(center[1] * sy)
#         cv2.circle(frame, (x, y), 5, (255, 0, 0), -1) 

#     return frame



def draw_box(frame: np.ndarray, color: Tuple[int, int, int], x: float, y: float, text: str, font_height: int) -> np.ndarray:
    """
    Draw a labeled box on an image.
    """
    x, y = int(x), int(y)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.rectangle(frame, (x - font_height // 2, y - font_height // 2), (x + font_height // 2, y + font_height // 2), color, -1)  # Filled box
    cv2.putText(frame, text, (x + 5, y - 5), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)  # White text

    return frame



def draw_points(canvas: np.ndarray, points: List[Tuple[float, float]]) -> np.ndarray:
    """
    Draws the centers as circles on an OpenCV image (canvas) and returns the modified frame.
    
    :param canvas: The image or frame to draw on.
    :param centers: List of (x, y) coordinates, shape (64, 2).
    :return: The modified frame with drawn centers.
    """
    
    frame_height, frame_width = canvas.shape[:2]
    
    for i, (x, y) in enumerate(points):  # Assuming shape (64, 2)
        x = round(x * frame_width / MODEL_WIDTH) 
        y = round(y * frame_height / MODEL_HEIGHT) 

        cv2.circle(canvas, (x, y), radius=5, color=(0, 0, 255), thickness=-1)

    return canvas



def draw_polygon(canvas, polygon):
    # Get the height and width of the canvas (image)
    frame_height, frame_width = canvas.shape[:2]
    
    # Define the scaling factors (assuming MODEL_WIDTH and MODEL_HEIGHT are defined elsewhere)
    sx = frame_width / MODEL_WIDTH
    sy = frame_height / MODEL_HEIGHT
    
    # Scale the polygon coordinates
    scaled_polygon = np.array([
        [(x * sx, y * sy) for x, y in polygon]
    ], dtype=np.int32)

    # Draw the polygon on the canvas (image)
    cv2.polylines(canvas, scaled_polygon, isClosed=True, color=(0, 0, 255), thickness=2)

    # Optionally, you can fill the polygon if you need:
    # cv2.fillPoly(canvas, scaled_polygon, color=(0, 255, 0))  # Green fill
    
    return canvas


# def render_state(canvas_ref, centers, state):
#     # Set up canvas context (this can be modified according to your specific setup)
#     ctx, font_height, line_width, sx, sy = setup_ctx(canvas_ref)

#     draw_points(ctx, centers, "blue", sx, sy)
    
#     for i in range(64):
#         best_score = 0.1
#         best_piece = -1

#         # Find the best piece for the current position
#         for j in range(12):
#             if state[i][j] > best_score:
#                 best_score = state[i][j]
#                 best_piece = j

#         if best_piece == -1:
#             continue

#         color = PALETTE[best_piece % len(PALETTE)]
#         text = f"{LABELS[best_piece]}:{round(100 * best_score)}"

#         draw_box(ctx, color, centers[i][0] * sx, centers[i][1] * sy, text, font_height, line_width)
        
        

# def setup_ctx(canvas_ref: np.ndarray) -> Tuple:
#     """
#     Set up the canvas context for drawing, including font size, line width, 
#     and scaling factors based on the canvas size.
    
#     Parameters:
#     - canvas_ref: The canvas (or image) where drawing will occur (using OpenCV).
    
#     Returns:
#     - ctx: The canvas reference for drawing.
#     - font_height: The height of the font to be used for text rendering.
#     - line_width: The line width for drawing shapes.
#     - sx: Scaling factor for width.
#     - sy: Scaling factor for height.
#     """
#     height, width = canvas_ref.shape[:2]
    
#     # Set alpha for transparency
#     # alpha = 0.8

#     # Initialize the drawing context (in OpenCV, it's essentially the canvas itself)
#     ctx = canvas_ref
    
#     # Clear the context (using OpenCV to fill the canvas with a white color)
#     ctx[:] = 255  # White background

#     # Set font size and style
#     font_size = max(int(max(width, height) / 40), 14)
#     font_height = font_size
#     # font = cv2.FONT_HERSHEY_SIMPLEX  # OpenCV's default font style

#     # Set line width
#     line_width = max(min(width, height) / 200, 2.5)

#     # Scaling factors for width and height
#     sx = width / MODEL_WIDTH
#     sy = height / MODEL_HEIGHT

#     return ctx, font_height, line_width, sx, sy
