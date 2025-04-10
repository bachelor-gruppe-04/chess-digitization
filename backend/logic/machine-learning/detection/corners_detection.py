import numpy as np
import tensorflow as tf
import onnxruntime as ort

from typing import Tuple, List, Dict, Optional
from utilities.constants import MODEL_WIDTH, MODEL_HEIGHT, MARKER_DIAMETER, CORNER_KEYS
from maths.quad_transformation import get_quads, score_quad, perspective_transform, clamp
from detection.detection_methods import get_input, get_boxes_and_scores, euclidean_distance, get_center_of_set_of_points, process_boxes_and_scores, get_xy

def predict_xcorners(image: np.ndarray, corner_ort_session: ort.InferenceSession):
    """
    Perform corner detection on the input image.

    Args:
        image (numpy.ndarray): The input image to detect corners.
        corner_ort_session: The ONNX Runtime InferenceSession object for the corner detection model.

    Returns:
        numpy.ndarray: Array of predicted corner points (coordinates and confidence scores).
    """

    # Run inference for corner detection
    model_inputs = corner_ort_session.get_inputs()
    model_outputs = corner_ort_session.get_outputs()

    predictions = corner_ort_session.run(
        output_names=[output.name for output in model_outputs],
        input_feed={model_inputs[0].name: image}
    )

    #After doing the inference we get a prediction of where the corners are in this image. The format
    #of the prediction is on the form float16[1,5,2835] where 1 is batch size, 5 represents the coordiantes of
    #the corners (4) and (1) for confidence score. Lastly the 2835 is the number of n_anchors or anchor boxes. Anchor
    #boxes are pre-defined boxes that the model uses to predict the corners. The image is a grid of 2835 tiny boxes 
    #where the model uses these to predict the corners
    xcorner_predictions = predictions[0]  

    return xcorner_predictions



async def run_xcorners_model(frame: np.ndarray, corners_model_ref: ort.InferenceSession, pieces: List[dict]) -> List[List[float]]:
    """
    Processes a video reference using a corners detection model to predict x_corners pieces in the video.

    Parameters:
    - frame: A reference to the video data.
    - corners_model_ref: A reference to the corner detection model used to predict the corners of the pieces.
    - pieces: A list of detected chess pieces, each containing information about their bounding box and class.

    Returns:
    - preds: A list of predicted x_corner positions
    """
    video_height, video_width, _ = frame.shape

    # Extract the keypoints (coordinates of the chess pieces) from the pieces list.
    # These will serve as input to the corner detection model to help refine predictions.
    keypoints: List[List[float]] = [[x[0], x[1]] for x in pieces]  # List of (x, y) positions of the pieces

    # Prepare the input image for the x_corner detection model, including keypoints as an additional input.
    image4d: np.ndarray
    width: int
    height: int
    padding: List[int]
    roi: List[int]
    image4d, width, height, padding, roi = get_input(frame, keypoints)

    # Run the x_corner detection model on the preprocessed image to get predictions.
    x_corner_predictions: tf.Tensor = predict_xcorners(image4d, corners_model_ref)

    # Extract the bounding boxes and scores from the x_corner predictions
    boxes: tf.Tensor
    scores: tf.Tensor
    boxes, scores = get_boxes_and_scores(x_corner_predictions, width, height, video_width, video_height, padding, roi)

    # Clean up intermediate variables to free up memory
    del x_corner_predictions 
    del image4d 
   
    # Process the boxes and scores using non-max suppression and other techniques
    # This helps to clean up redundant boxes and refine the corner detection.
    # Should return 49 x_corners (7x7 grid)
    x_corners_optimized: np.ndarray = process_boxes_and_scores(boxes, scores)

    # Extracts the x and y values from x_corners_optimized 
    x_corners: List[List[float]] = [[x[0], x[1]] for x in x_corners_optimized]

    return x_corners



def find_board_corners_from_xcorners(x_corners: np.ndarray) -> Optional[List[List[float]]]:
    """
    Given the detected x_corners, find the corners of a quadrilateral using perspective transformation.
    
    Parameters:
    - x_corners: A NumPy array representing the optimized x_corner predictions (should be of shape (N, 2)).
    
    Returns:
    - A list of four corners (each represented as [x, y] coordinates) if a valid quadrilateral is found.
    - None if no valid quadrilateral could be determined.
    """
    quads: List[np.ndarray] = get_quads(x_corners)
    
    if len(quads) == 0:
        return None
    
    best_score: float = None
    best_m: np.ndarray = None
    best_offset: np.ndarray = None
    
    # Assume score_quad returns score, M, offset
    best_score, best_m, best_offset = score_quad(quads[0], x_corners)
    
    for quad in quads[1:]:
        score, m, offset = score_quad(quad, x_corners)
        if score > best_score:
            best_score = score
            best_m = m
            best_offset = offset
    
    # Inverse matrix calculation
    inv_m: np.ndarray = np.linalg.inv(best_m)
    
    # Define warped corners based on offset
    warped_corners: List[List[float]] = [
        [best_offset[0] - 1, best_offset[1] - 1],
        [best_offset[0] - 1, best_offset[1] + 7],
        [best_offset[0] + 7, best_offset[1] + 7],
        [best_offset[0] + 7, best_offset[1] - 1]
    ]
    
    # Apply perspective transform
    corners: List[List[float]] = perspective_transform(warped_corners, inv_m)
    
    # Clip corners
    for i in range(4):
        corners[i][0] = clamp(corners[i][0], 0, MODEL_WIDTH)
        corners[i][1] = clamp(corners[i][1], 0, MODEL_HEIGHT)
    
    return corners



def assign_labels_to_board_corners(black_pieces: List[np.ndarray], white_pieces: List[np.ndarray], corners: List[List[float]]) -> Dict[str, List[float]]:
    """
    Assigns labels (a1, h1, h8, a8) to the corners of a chessboard based on the positions of the black and white pieces.

    Parameters:
    - black_pieces: A list of NumPy arrays representing the positions of the black pieces (each element should be an [x, y] coordinate).
    - white_pieces: A list of NumPy arrays representing the positions of the white pieces (each element should be an [x, y] coordinate).
    - corners: A list of four lists of floats representing the coordinates of the corners of the chessboard.

    Returns:
    - A dictionary mapping board labels (e.g., "a1", "h1", "h8", "a8") to the corresponding corner coordinates.
    """
    black_center: List[float] = get_center_of_set_of_points(black_pieces)
    white_center: List[float] = get_center_of_set_of_points(white_pieces)
    
    best_shift: int = 0
    best_score: float = 0
    for shift in range(4):
        cw: List[float] = [(corners[shift % 4][0] + corners[(shift + 1) % 4][0]) / 2,
                           (corners[shift % 4][1] + corners[(shift + 1) % 4][1]) / 2]
        cb: List[float] = [(corners[(shift + 2) % 4][0] + corners[(shift + 3) % 4][0]) / 2,
                           (corners[(shift + 2) % 4][1] + corners[(shift + 3) % 4][1]) / 2]
        score: float = 1 / (1 + euclidean_distance(white_center, cw) + euclidean_distance(black_center, cb))
        if score > best_score:
            best_score = score
            best_shift = shift

    keypoints: Dict[str, List[float]] = {
        "h1": corners[best_shift % 4],
        "a1": corners[(best_shift + 1) % 4],
        "a8": corners[(best_shift + 2) % 4],
        "h8": corners[(best_shift + 3) % 4]
    }
    
    return keypoints




def extract_xy_from_labeled_corners(corners_mapping: Dict[str, Dict[str, Tuple[int, int]]], canvas_ref: np.ndarray) -> List[Tuple[int, int]]:
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



def scale_xy_board_corners(xy: Tuple[float, float], height: int, width: int) -> List[float]:
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