
import numpy as np
import tensorflow as tf

from typing import Tuple, List, Dict, Optional
from constants import MODEL_WIDTH, MODEL_HEIGHT, MARKER_DIAMETER, CORNER_KEYS
from preprocess import preprocess_image

def process_boxes_and_scores(boxes: tf.Tensor, scores: tf.Tensor) -> np.ndarray:
    """
    Processes bounding boxes and scores to apply non-max suppression (NMS),
    extract centers of selected boxes, and concatenate them with their class indices.

    Args:
        boxes (tf.Tensor): A tensor of shape (num_boxes, 4), representing bounding box coordinates [x_min, y_min, x_max, y_max].
        scores (tf.Tensor): A tensor of shape (num_boxes, num_classes), representing the classification scores for each box.

    Returns:
        np.ndarray: A NumPy array containing the centers of the selected bounding boxes and their corresponding class indices.
    """
    max_scores: tf.Tensor = tf.reduce_max(scores, axis=1)
    argmax_scores: tf.Tensor = tf.argmax(scores, axis=1)
    nms: tf.Tensor = tf.image.non_max_suppression(boxes, max_scores, max_output_size=100, iou_threshold=0.3, score_threshold=0.1)
    
    # Use get_centers function to get the centers from the selected boxes
    centers_bbox: tf.Tensor = get_centers_of_bbox(tf.gather(boxes, nms, axis=0))

    # Gather the class indices of the selected boxes and expand dimensions
    class_indices: tf.Tensor = tf.expand_dims(tf.gather(argmax_scores, nms, axis=0), axis=1)

    # Cast cls to float16 (ensure it's compatible with centers)
    class_indices: tf.Tensor = tf.cast(class_indices, dtype=tf.float16)

    # Concatenate the centers with the class indices
    res: tf.Tensor = tf.concat([centers_bbox, class_indices], axis=1)

    res_array: np.ndarray = res.numpy()
    
    return res_array



def get_boxes_and_scores(
    preds: np.ndarray, 
    width: int, 
    height: int, 
    video_width: int, 
    video_height: int, 
    padding: Tuple[int, int, int, int], 
    roi: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    This function processes predictions to extract bounding boxes and their associated scores.

    Args:
        preds (np.ndarray): The predictions array with shape (batch_size, num_boxes, num_predictions).
        width (int): The width to scale the bounding box coordinates to.
        height (int): The height to scale the bounding box coordinates to.
        video_width (int): The width of the video for further scaling.
        video_height (int): The height of the video for further scaling.
        padding (Tuple[int, int, int, int]): The padding to adjust the bounding boxes.
        roi (Tuple[int, int]): The region of interest, which is added to the bounding box coordinates.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the bounding boxes and scores.
    """
    
    # Transpose preds to match the desired shape
    preds_t: np.ndarray = np.transpose(preds, (0, 2, 1))  # Shape: (batch_size, num_predictions, num_boxes)
    
    # Extract width (w) and height (h) of the boxes
    w: np.ndarray = preds_t[:, :, 2:3]  
    h: np.ndarray = preds_t[:, :, 3:4]  
    
    # Convert xc, yc, w, h to l, t, r, b (left, top, right, bottom)
    l: np.ndarray = preds_t[:, :, 0:1] - (w / 2)  # Left
    t: np.ndarray = preds_t[:, :, 1:2] - (h / 2)  # Top
    r: np.ndarray = l + w  # Right
    b: np.ndarray = t + h  # Bottom

    # Apply padding to the bounding box coordinates
    l -= padding[0]
    r -= padding[0]
    t -= padding[2]
    b -= padding[2]

    # Scale the bounding box coordinates to the target size
    l *= (width / (MODEL_WIDTH - padding[0] - padding[1]))
    r *= (width / (MODEL_WIDTH - padding[0] - padding[1]))
    t *= (height / (MODEL_HEIGHT - padding[2] - padding[3]))
    b *= (height / (MODEL_HEIGHT - padding[2] - padding[3]))

    # Add ROI (Region of Interest)
    l += roi[0]
    r += roi[0]
    t += roi[1]
    b += roi[1]

    # Scale based on video size
    l *= (MODEL_WIDTH / video_width)
    r *= (MODEL_WIDTH / video_width)
    t *= (MODEL_HEIGHT / video_height)
    b *= (MODEL_HEIGHT / video_height)

    # Concatenate the left, top, right, and bottom coordinates to form the bounding boxes
    boxes: np.ndarray = np.concatenate([l, t, r, b], axis=2) 

    # Extract the scores (assuming score is in the 5th element onward)
    scores: np.ndarray = preds_t[:, :, 4:] 

    # Squeeze to remove unnecessary dimensions (if any)
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)

    return boxes, scores



def get_center(points: List[Tuple[float, float]]) -> List[float]:
    """
    Calculates the center of a set of 2D points by averaging their x and y coordinates.

    Args:
        points (List[Tuple[float, float]]): A list of points where each point is a tuple of (x, y) coordinates.

    Returns:
        List[float]: A list containing the (x, y) coordinates of the center.
    """
    center_x: float = sum(x[0] for x in points)
    center_y: float = sum(x[1] for x in points) 
    center: List[float] = [center_x / len(points), center_y / len(points)]
    
    return center



def euclidean(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """
    Calculates the Euclidean distance between two points in 2D space.

    Args:
        a (Tuple[float, float]): The first point as a tuple of (x, y).
        b (Tuple[float, float]): The second point as a tuple of (x, y).

    Returns:
        float: The Euclidean distance between the two points.
    """
    dx: float = a[0] - b[0]  # Difference in x-coordinates
    dy: float = a[1] - b[1]  # Difference in y-coordinates
    
    # Calculate the Euclidean distance
    dist: float = (dx ** 2 + dy ** 2) ** 0.5
    
    return dist



def get_input(
    video_ref: np.ndarray, 
    keypoints: Optional[np.ndarray] = None, 
    padding_ratio: int = 12
) -> Tuple[int, int, Optional[np.ndarray]]:
    """
    Processes the input video frame by extracting the region of interest (ROI),
    resizing it to match the model's input dimensions, and applying necessary padding.

    Args:
        video_ref (np.ndarray): Input video frame represented as a NumPy array of shape (height, width, channels).
        keypoints (Optional[np.ndarray]): Array of keypoints used to determine the bounding box (ROI). Defaults to None.
        padding_ratio (int): Factor to compute padding around the detected bounding box. Defaults to 12.

    Returns:
        Tuple[np.ndarray, int, int, List[int], List[int]]:
            - image4d (np.ndarray): Preprocessed and padded input ready for the model.
            - width (int): Width of the cropped video region.
            - height (int): Height of the cropped video region.
            - padding (List[int]): Padding applied as [left, right, top, bottom].
            - roi (List[int]): Coordinates of the region of interest in the format [xmin, ymin, xmax, ymax].
    """

    video_height, video_width, _ = video_ref.shape
    roi = None

    if keypoints is not None:
        bbox = get_bbox(keypoints)
        padding_left = int(bbox['width'] / padding_ratio)
        padding_right = int(bbox['width'] / padding_ratio)
        padding_top = int(bbox['height'] / padding_ratio)
        padding_bottom = int(bbox['height'] / padding_ratio)

        padded_roi_width = bbox['width'] + padding_left + padding_right
        padded_roi_height = bbox['height'] + padding_top + padding_bottom
        ratio = padded_roi_height / padded_roi_width
        desired_ratio = MODEL_HEIGHT / MODEL_WIDTH

        if ratio > desired_ratio:
            target_width = padded_roi_height / desired_ratio
            dx = target_width - padded_roi_width
            padding_left += int(dx / 2)
            padding_right += dx - int(dx / 2)
        else:
            target_height = padded_roi_width * desired_ratio
            padding_top += target_height - padded_roi_height

        roi = [
            max(int(video_width * (bbox['xmin'] - padding_left) / MODEL_WIDTH), 0),
            max(int(video_height * (bbox['ymin'] - padding_top) / MODEL_HEIGHT), 0),
            min(int(video_width * (bbox['xmax'] + padding_right) / MODEL_WIDTH), video_width),
            min(int(video_height * (bbox['ymax'] + padding_bottom) / MODEL_HEIGHT), video_height)
        ]
    else:
        roi = [0, 0, video_width, video_height]

    # Cropping
    video_ref = video_ref[roi[1]:roi[3], roi[0]:roi[2], :]
    
    # Resizing
    height, width, _ = video_ref.shape
    ratio = height / width
    desired_ratio = MODEL_HEIGHT / MODEL_WIDTH
    resize_height = MODEL_HEIGHT
    resize_width = MODEL_WIDTH
    if ratio > desired_ratio:
        resize_width = int(MODEL_HEIGHT / ratio)
    else:
        resize_height = int(MODEL_WIDTH * ratio)
    
    video_ref = tf.image.resize(video_ref, [resize_height, resize_width])
    
    # Padding
    dx = MODEL_WIDTH - video_ref.shape[1]
    dy = MODEL_HEIGHT - video_ref.shape[0]
    pad_right = dx // 2
    pad_left = dx - pad_right
    pad_bottom = dy // 2
    pad_top = dy - pad_bottom
    padding = [pad_left, pad_right, pad_top, pad_bottom]

    image4d = preprocess_image(video_ref.numpy())
    
    return image4d, width, height, padding, roi



def extract_xy_from_corners_mapping(corners_mapping: Dict[str, Dict[str, Tuple[int, int]]], canvas_ref: np.ndarray) -> List[Tuple[int, int]]:
    """
    Extracts normalized (x, y) coordinates from a given corners mapping and a canvas reference.

    Args:
        corners_mapping (Dict[str, Dict[str, Tuple[int, int]]]): A mapping of corner names to their respective 'xy' coordinates.
        canvas_ref (np.ndarray): A reference to the canvas as a numpy array, representing the image or drawing.

    Returns:
        List[Tuple[int, int]]: A list of (x, y) coordinates corresponding to each corner in the mapping.
    """
    canvas_height, canvas_width, _ = canvas_ref.shape 
    return [get_xy(corners_mapping[x]['xy'], canvas_height, canvas_width) for x in CORNER_KEYS]



def get_xy(marker_xy: Tuple[int, int], height: int, width: int) -> Tuple[float, float]:
    """
    Converts marker coordinates to a normalized system based on canvas dimensions.

    Args:
        marker_xy (Tuple[int, int]): The (x, y) coordinates of the marker.
        height (int): The height of the canvas.
        width (int): The width of the canvas.

    Returns:
        Tuple[float, float]: The normalized (x, y) coordinates of the marker.
    """
    sx: float = MODEL_WIDTH / width 
    sy: float = MODEL_HEIGHT / height
    xy: Tuple[float, float] = (sx * marker_xy[0], sy * (marker_xy[1] + height + MARKER_DIAMETER))
    return xy



def get_bbox(points: List[Tuple[float, float]]) -> Dict[str, float]:
    """
    Computes the bounding box for a list of 2D points.

    The bounding box is the smallest rectangle that can contain all the points, 
    with sides aligned to the axes. It is defined by the minimum and maximum 
    x and y coordinates of the points.

    Args:
        points (List[Tuple[float, float]]): A list of 2D points, where each point 
                                             is represented as a tuple (x, y).

    Returns:
        Dict[str, float]: A dictionary containing the bounding box parameters:
                          "xmin", "xmax", "ymin", "ymax", "width", and "height".
    """
    xs: List[float] = [p[0] for p in points]
    ys: List[float] = [p[1] for p in points]
    
    xmin: float = min(xs)
    xmax: float = max(xs)
    ymin: float = min(ys)
    ymax: float = max(ys)

    width: float = xmax - xmin
    height: float = ymax - ymin

    bbox: Dict[str, float] = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": width,
        "height": height
    }

    return bbox



def scale_labeled_board_corners(xy: Tuple[float, float], height: int, width: int) -> List[float]:
    """
    Scales the (x, y) coordinates of a labeled board marker to fit within the canvas size.

    This function adjusts the coordinates of the marker to match the scaling of the 
    canvas, accounting for the difference between the model's size and the canvas's size.

    Args:
        xy (Tuple[float, float]): The (x, y) coordinates of the marker in the model's coordinate system.
        height (int): The height of the canvas.
        width (int): The width of the canvas.

    Returns:
        List[float]: A list containing the scaled (x, y) coordinates of the marker in the canvas coordinate system.
    """
    sx: float = width / MODEL_WIDTH
    sy: float = height / MODEL_HEIGHT
    marker_xy: List[float] = [sx * xy[0], sy * xy[1] - height - MARKER_DIAMETER]
    
    return marker_xy



def get_centers_of_bbox(boxes: tf.Tensor) -> tf.Tensor:
    """
    Calculates the center coordinates of bounding boxes.

    This function computes the center (cx, cy) of each bounding box from 
    the left, top, right, and bottom coordinates. It assumes that the input 
    boxes are in the format [left, top, right, bottom] and casts them to `float16`
    for consistency with the model's data type.

    Args:
        boxes (tf.Tensor): A tensor of shape (N, 4), where each row represents 
                            a bounding box with the format [left, top, right, bottom].

    Returns:
        tf.Tensor: A tensor of shape (N, 2), where each row contains the center 
                   coordinates [cx, cy] of the corresponding bounding box.
    """
    # Ensure boxes are of type float16 (as your model is using float16)
    boxes = tf.cast(boxes, dtype=tf.float16)

    # Extract left, top, right, and bottom coordinates
    l = boxes[:, 0:1]
    t = boxes[:, 1:2]
    r = boxes[:, 2:3]
    b = boxes[:, 3:4]

    # Calculate center coordinates (cx, cy)
    cx = (l + r) / 2
    cy = (t + b) / 2

    centers = tf.concat([cx, cy], axis=1)
    
    return centers


