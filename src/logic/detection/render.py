import cv2
import matplotlib.pyplot as plt
import numpy as np
from constants import PALETTE, LABELS


# def render_corners(video_frame, xCorners, target_width=480, target_height=288, video_width=1920, video_height=1080):
#     """
#     Renders corners on the frame by calling draw_points.

#     Args:
#     - frame: The image/frame to render the corners on.
#     - xCorners: List of corner coordinates to draw on the frame.
#     - target_width: The width of the model input (default is 480).
#     - target_height: The height of the model input (default is 288).
#     - video_width: The width of the original video frame (default is 1920).
#     - video_height: The height of the original video frame (default is 1080).
    
#     Returns:
#     - frame_with_corners: The image/frame with corners rendered.
#     """
#     # Calculate scaling factors for width and height
#     scale_x = video_width / target_width
#     scale_y = video_height / target_height

#     # Use draw_points to render the corners with the scaling factors
#     frame_with_corners = draw_points1(video_frame.copy(), xCorners, "blue", scale_x, scale_y)

#     return frame_with_corners



# def draw_points1(frame, points, color="blue", scale_x=1, scale_y=1):
#     """
#     Draw points on the frame.
    
#     Args:
#     - frame: The image/frame to draw on.
#     - points: A list of (x, y) coordinates to draw on the frame.
#     - color: Color to draw the points (default is blue).
#     - scale_x: Scaling factor for x-axis coordinates.
#     - scale_y: Scaling factor for y-axis coordinates.
#     """
#     # Convert color name to BGR tuple (since OpenCV uses BGR format)
#     color_map = {
#         "blue": (255, 0, 0),
#         "green": (0, 255, 0),
#         "red": (0, 0, 255)
#     }
#     color = color_map.get(color, (255, 0, 0))  # Default to blue if invalid color

#     for point in points:
#         # Scale the points (if needed)
#         x = int(point[0] * scale_x)
#         y = int(point[1] * scale_y)

#         # Draw a circle at each corner
#         cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blå farge i BGR-format


#     return frame

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


import cv2

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
    # Get the dimensions of the canvas/frame



    # Calculate scaling factors
    scale_x = 1920 / 480
    scale_y = 1080 / 288

    # for (x, y) in centers:
    #     # Apply scaling to coordinates
    #     scaled_x = int(x * 1)
    #     scaled_y = int(y * scal1e_y)

    #     # Draw a red circle for each center, applying the scaling
    #     cv2.circle(canvas, (scaled_x, scaled_y), 5, (0, 0, 255), -1)  # Red dot for each center


    
    for center in centers:
        # Scale the points (if needed)
        x = int(center[0] * 2)
        y = int(center[1] * 2)

        print(x, y)

        # Draw a circle at each corner
        cv2.circle(canvas, (x, y), 5, (255, 0, 0), -1)  # Blå farge i BGR-format

    
    return canvas  # Return the modified image


def render_state(frame, centers):
    """
    Render the entire state onto the frame using OpenCV.
    """
    # sx, sy = 10, 10  # Scaling factors
    # font_height = 12
    # line_width = 2

    new_frame = visualize_centers_opencv(frame, centers)



    # draw_polygon(frame, boundary, (255, 0, 0), sx, sy)  # Blue polygon
    
    # for i in range(64):
    #     best_score = 0.1
    #     best_piece = -1

        # Simulated state check (replace with actual logic)
        # for j in range(12):
        #     if state[i][j] > best_score:
        #         best_score = state[i][j]
        #         best_piece = j

        # if best_piece == -1:
        #     continue
        
        # color = PALETTE[best_piece % len(PALETTE)]
        # text = f"{LABELS[best_piece]}:{round(100 * best_score)}"

        # draw_box(frame, color, centers[i][0] * sx, centers[i][1] * sy, text, font_height, line_width)

    return new_frame