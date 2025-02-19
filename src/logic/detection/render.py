import cv2

def render_corners(frame, xCorners, target_width=480, target_height=288, video_width=1920, video_height=1080):
    """
    Renders corners on the frame by calling draw_points.

    Args:
    - frame: The image/frame to render the corners on.
    - xCorners: List of corner coordinates to draw on the frame.
    - target_width: The width of the model input (default is 480).
    - target_height: The height of the model input (default is 288).
    - video_width: The width of the original video frame (default is 1920).
    - video_height: The height of the original video frame (default is 1080).
    
    Returns:
    - frame_with_corners: The image/frame with corners rendered.
    """
    # Calculate scaling factors for width and height
    scale_x = video_width / target_width
    scale_y = video_height / target_height

    # Use draw_points to render the corners with the scaling factors
    frame_with_corners = draw_points(frame.copy(), xCorners, "blue", scale_x, scale_y)
    
    return frame_with_corners



def draw_points(frame, points, color="blue", scale_x=1, scale_y=1):
    """
    Draw points on the frame.
    
    Args:
    - frame: The image/frame to draw on.
    - points: A list of (x, y) coordinates to draw on the frame.
    - color: Color to draw the points (default is blue).
    - scale_x: Scaling factor for x-axis coordinates.
    - scale_y: Scaling factor for y-axis coordinates.
    """
    # Convert color name to BGR tuple (since OpenCV uses BGR format)
    color_map = {
        "blue": (255, 0, 0),
        "green": (0, 255, 0),
        "red": (0, 0, 255)
    }
    color = color_map.get(color, (255, 0, 0))  # Default to blue if invalid color

    for point in points:
        # Scale the points (if needed)
        x = int(point[0] * scale_x)
        y = int(point[1] * scale_y)

        # Draw a circle at each corner
        cv2.circle(frame, (x, y), 5, color, -1)  # -1 for filled circle

    return frame