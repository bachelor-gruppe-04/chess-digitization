import numpy as np
import onnxruntime as ort

from typing import List, Dict, Tuple, Optional
from logic.machine_learning.utilities.constants import CORNER_KEYS
from logic.machine_learning.detection.corners_detection import run_xcorners_model, find_board_corners_from_xcorners, assign_labels_to_board_corners, scale_xy_board_corners, extract_xy_from_labeled_corners
from logic.machine_learning.detection.piece_detection import run_pieces_model
from logic.machine_learning.maths.warp import get_inv_transform, transform_centers, transform_boundary

async def get_board_corners(video_ref: np.ndarray, pieces_model_ref: ort.InferenceSession, xcorners_model_ref: ort.InferenceSession) -> Optional[np.ndarray]: 
    """
    Detects corners on a chessboard using ONNX models.

    Args:
        video_ref (np.ndarray): The input video frame.
        pieces_model_ref (ort.InferenceSession): ONNX model for detecting chess pieces.
        xcorners_model_ref (ort.InferenceSession): ONNX model for detecting x_corners.

    Returns:
        Optional[np.ndarray]: Processed frame with centers visualized, or None if detection fails.
    """
    video_height, video_width, _ = video_ref.shape

    # We extract the top 16 predicted pieces for both black and white players
    # Pieces is on the format [x, y, pieceTypeIndex]
    
    pieces: List[List[int]] = await run_pieces_model(video_ref, pieces_model_ref)

    # Metadata of model tells us white pieces index range from 0-5 
    # while black pieces index from 6-11
    black_pieces: List[List[int]] = [x for x in pieces if x[2] <= 5]
    white_pieces: List[List[int]] = [x for x in pieces if x[2] > 5]

    if len(black_pieces) == 0 or len(white_pieces) == 0:
        print("Black or white pieces not detected")
        return None

    # Extracts the top 49 predicted x_corners for the chess board (inner 7x7 grid)
    x_corners: List[List[int]] = await run_xcorners_model(video_ref, xcorners_model_ref, pieces)

    if len(x_corners) < 5:
        print("Not enough x_corners")
        return None

    # Extracts the 4 outer corners of the chess board
    # Important to note that these board_corners ARE NOT labeled (a1,a8,h1,h8)
    board_corners: List[Tuple[int, int]] = find_board_corners_from_xcorners(x_corners)

    # Assigns the labels (a1,a8,h1,h8) to the board_corners based on the 
    # placement of the white and black pieces
    labeled_board_corners: Dict[str, Tuple[int, int]] = assign_labels_to_board_corners(black_pieces, white_pieces, board_corners)

    scaled_labeled_board_corners: Dict[str, Dict[str, Tuple[int, int]]] = {}

    for key in CORNER_KEYS:
        xy: Tuple[int, int] = labeled_board_corners[key]
        payload: Dict[str, Tuple[int, int] | str] = {
            "xy": scale_xy_board_corners(xy, video_height, video_width),
            "key": key
        }
        scaled_labeled_board_corners[key] = payload 
    
    return scaled_labeled_board_corners


def find_centers_and_boundary(corners_mapping: Dict[str, Dict[str, Tuple[int, int]]], frame: np.ndarray) -> List[List[Tuple[float, float]]]:
    """
    Finds the centers of squares from video frame.

    First extracts the coordinates of the corner points from the `corners_mapping` and `frame`, 
    then computes the inverse transformation matrix to map these corner points from the distorted image back to 
    the perfect square grid. Finally, it calculates and returns the centers of the squares in the transformed space.

    Args:
    - corners_mapping (dict): A mapping of the chessboard corners (or other keypoints) 
      extracted from the video reference.
    - frame (np.ndarray): The video frame or reference image used for extracting the corners.

    Returns:
    - centers (list): The transformed center coordinates of the squares after applying the inverse perspective transformation.
    """

    xy: List[Tuple[int, int]] = extract_xy_from_labeled_corners(corners_mapping, frame)
    inv_transform: np.ndarray = get_inv_transform(xy)
    centers, centers3D = transform_centers(inv_transform)
    boundary, boundary3D = transform_boundary(inv_transform)
        
    return centers, boundary, centers3D, boundary3D
