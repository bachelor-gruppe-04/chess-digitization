import numpy as np
import tensorflow as tf
import time
from typing import List, Tuple

from detection.piece_detection import detect
from detection.corners_detection import extract_xy_from_labeled_corners
from game.game import make_update_payload
from view.render import draw_points, draw_polygon, draw_boxes_with_scores
from detection.run_detections import find_centers_and_boundary

last_update_time = 0  # Global or external to function if needed

async def get_payload(
    piece_model_ref: tf.keras.Model,
    video_ref: np.ndarray,
    corners_ref: np.ndarray,
    game_ref: any,
    moves_pairs_ref: list
) -> tuple:
    """
    Main function to generate the payload for the game. It computes the state of the game based on the current video frame,
    updates the state of the board, processes possible moves, and prepares the payload for dispatch.

    Args:
        piece_model_ref (tf.keras.Model): The model used for piece detection.
        video_ref (np.ndarray): The video frame from the game.
        corners_ref (np.ndarray): Coordinates of the corners in the video frame.
        game_ref (any): The game object, representing the current state of the game.
        moves_pairs_ref (list): List of possible move pairs in the game.

    Returns:
        tuple: A tuple containing the updated video frame and the game payload.
    """
    global last_update_time

    centers = None
    boundary = None
    centers_3d = None
    boundary_3d = None
    state = None
    payload = None
    keypoints = None
    possible_moves = set()
    greedy_move_to_time = {}

    if centers is None:
        keypoints = extract_xy_from_labeled_corners(corners_ref, video_ref)
        centers, boundary, centers_3d, boundary_3d = find_centers_and_boundary(corners_ref, video_ref)
        state = np.zeros((64, 12))
        possible_moves = set()
        greedy_move_to_time = {}

    start_time = time.time()
    
    boxes, scores = await detect(piece_model_ref, video_ref, keypoints)

    squares = get_squares(boxes, centers_3d, boundary_3d)

    current_time = time.time()
    if current_time - last_update_time >= 4.0:
        update = get_update(scores, squares)
        last_update_time = current_time
    else:
        update = np.zeros((64, 12))  # No update this frame

    state = update_state(state, update)
    best_score1, best_score2, best_joint_score, best_move, best_moves = process_state(
        state, moves_pairs_ref, possible_moves
    )

    end_time = time.time()
    print("FPS:", round(1 / (end_time - start_time), 1))

    has_move = False
    if best_moves is not None:
        move_str = best_moves["sans"][0]
        has_move = best_score2 > 0 and best_joint_score > 0 and move_str in possible_moves
        if has_move:
            game_ref.board.push_san(move_str)
            possible_moves.clear()
            greedy_move_to_time = {}

    has_greedy_move = False
    if best_move is not None and not has_move and best_score1 > 0:
        move_str = best_move["sans"][0]
        if move_str not in greedy_move_to_time:
            greedy_move_to_time[move_str] = end_time

        elapsed = (end_time - greedy_move_to_time[move_str]) > 1
        is_new = san_to_lan(game_ref.board, move_str) != game_ref.last_move
        has_greedy_move = elapsed and is_new
        if has_greedy_move:
            game_ref["board"].move(move_str)
            greedy_move_to_time = {move_str: greedy_move_to_time[move_str]}

    if has_move or has_greedy_move:
        payload = make_update_payload(game_ref.board, greedy=False)
        # dispatch(game_update(payload))
        
    draw_points(video_ref, centers)
    draw_polygon(video_ref, boundary)
    # draw_boxes_with_scores(video_ref, boxes, scores, threshold=0.5)

    tf.keras.backend.clear_session()
    
    return video_ref, payload


def calculate_move_score(state: np.ndarray, move: dict, from_thr: float = 0.6, to_thr: float = 0.6) -> float:
    """
    Calculates the score for a given move based on the state of the game.

    Args:
        state (np.ndarray): The current game state (64x12 array).
        move (dict): A dictionary containing information about the move ('from', 'to', and 'targets').
        from_thr (float): Threshold for scoring the 'from' squares. Default is 0.6.
        to_thr (float): Threshold for scoring the 'to' squares. Default is 0.6.

    Returns:
        float: The calculated score for the move.
    """
    score = 0
    for square in move['from']:
        score += 1 - max(state[square]) - from_thr
    
    for i in range(len(move['to'])):
        score += state[move['to'][i]][move['targets'][i]] - to_thr
    
    return score


def process_state(state: np.ndarray, moves_pairs: list, possible_moves: set) -> tuple:
    """
    Processes the game state and determines the best possible moves.

    Args:
        state (np.ndarray): The current game state (64x12 array).
        moves_pairs (list): List of possible move pairs.
        possible_moves (set): A set of possible moves.

    Returns:
        tuple: A tuple containing the best scores and moves.
    """
    best_score1 = float('-inf')
    best_score2 = float('-inf')
    best_joint_score = float('-inf')
    best_move = None
    best_moves = None
    seen = set()
    
    for move_pair in moves_pairs:
        if move_pair['move1']['sans'][0] not in seen:
            seen.add(move_pair['move1']['sans'][0])
            score = calculate_move_score(state, move_pair['move1'])
            
            if score > 0:
                possible_moves.add(move_pair['move1']['sans'][0])

            if score > best_score1:
                best_move = move_pair['move1']
                best_score1 = score
        
        if move_pair['move2'] is None or move_pair['moves'] is None or move_pair['move1']['sans'][0] not in possible_moves:
            continue
        
        score2 = calculate_move_score(state, move_pair['move2'])
        if score2 < 0:
            continue
        if score2 > best_score2:
            best_score2 = score2
        
        joint_score = calculate_move_score(state, move_pair['moves'])
        if joint_score > best_joint_score:
            best_joint_score = joint_score
            best_moves = move_pair['moves']
    
    return best_score1, best_score2, best_joint_score, best_move, best_moves


def get_box_centers(boxes: tf.Tensor) -> tf.Tensor:
    """
    Calculates the center coordinates (cx, cy) of bounding boxes.

    Args:
        boxes (tf.Tensor): A tensor of shape (N, 4), where each row represents 
                           a bounding box in the format [left, top, right, bottom].

    Returns:
        tf.Tensor: A tensor of shape (N, 2), where each row contains the center 
                   coordinates [cx, cy] of the corresponding bounding box.
    """
    # Slice the boxes tensor to get l, r, and b
    l = tf.cast(boxes[:, 0:1], tf.float32)  # Ensure l is float32
    r = tf.cast(boxes[:, 2:3], tf.float32)  # Ensure r is float32
    b = tf.cast(boxes[:, 3:4], tf.float32)  # Ensure b is float32

    # Calculate the center coordinates
    cx = (l + r) / 2
    cy = b - (r - l) / 3

    # Concatenate cx and cy to get the box centers
    box_centers = tf.concat([cx, cy], axis=1)

    return box_centers

def get_squares(boxes: tf.Tensor, centers3D: tf.Tensor, boundary3D: tf.Tensor) -> tf.Tensor:
    """
    Given the boxes, centers, and boundary, computes the square for each box by 
    determining the index of the minimum distance between box centers and the provided centers.

    Args:
        boxes (tf.Tensor): A tensor of shape (N, 6) representing N 3D boxes. Each box is 
                            represented by its (xmin, ymin, zmin, xmax, ymax, zmax).
        centers3D (tf.Tensor): A tensor of shape (M, 3) representing M 3D centers.
        boundary3D (tf.Tensor): A tensor of shape (1, 4, 2) representing boundary coordinates for the boxes.

    Returns:
        tf.Tensor: A tensor of shape (N,) representing the square (index) for each box based on the distance to centers3D.
    """
    with tf.device('/CPU:0'):
        # Get the box centers
        box_centers_3D = tf.expand_dims(get_box_centers(boxes), 1)

        # Calculate distances
        dist = tf.reduce_sum(tf.square(box_centers_3D - centers3D), axis=2)

        # Get squares by finding the index of minimum distances
        squares = tf.argmin(dist, axis=1)

        # Shift the boundary3D tensor
        shifted_boundary_3D = tf.concat([
            tf.slice(boundary3D, [0, 1, 0], [1, 3, 2]),
            tf.slice(boundary3D, [0, 0, 0], [1, 1, 2]),
        ], axis=1)

        n_boxes = tf.shape(box_centers_3D)[0]

        # Calculate a, b, c, and d tensors
        a = tf.squeeze(tf.subtract(
            tf.slice(boundary3D, [0, 0, 0], [1, 4, 1]),
            tf.slice(shifted_boundary_3D, [0, 0, 0], [1, 4, 1])
        ), axis=2)

        b = tf.squeeze(tf.subtract(
            tf.slice(boundary3D, [0, 0, 1], [1, 4, 1]),
            tf.slice(shifted_boundary_3D, [0, 0, 1], [1, 4, 1])
        ), axis=2)

        c = tf.squeeze(tf.subtract(
            tf.slice(box_centers_3D, [0, 0, 0], [n_boxes, 1, 1]),
            tf.slice(shifted_boundary_3D, [0, 0, 0], [1, 4, 1])
        ), axis=2)

        d = tf.squeeze(tf.subtract(
            tf.slice(box_centers_3D, [0, 0, 1], [n_boxes, 1, 1]),
            tf.slice(shifted_boundary_3D, [0, 0, 1], [1, 4, 1])
        ), axis=2)

        # Calculate determinant
        det = tf.subtract(tf.multiply(a, d), tf.multiply(b, c))

        # Apply tf.where condition for negative det values
        # Determinant is negative for all values so new_squares isn't used
        new_squares = tf.where(
            tf.reduce_any(tf.less(det, 0), axis=1),  # Check if any det < 0 along axis 1
            tf.constant(-1, dtype=squares.dtype),    # Replace with -1
            squares                                   # Otherwise, keep original squares
        )

        return squares

def get_update(scores_tensor: tf.Tensor, squares: tf.Tensor) -> np.ndarray:
    """
    Given a tensor of scores and squares, this function groups the scores based on the square indices,
    and computes the maximum value for each group to update the state.

    Args:
        scores_tensor (tf.Tensor): A tensor of shape (N, 12) containing scores for each box.
        squares (tf.Tensor): A tensor of shape (N,) containing square indices for each box.

    Returns:
        np.ndarray: An array of shape (64, 12) where each row corresponds to the maximum score for that square.
    """
    scores = scores_tensor.numpy()
    update = np.zeros((64, 12))

    grouped = {i: [] for i in range(64)}

    for i, square in enumerate(squares):
        square = int(square)
        if square != -1:
            grouped[square].append(scores[i])

    for square, group in grouped.items():
        if group:  # skip empty
            update[square] = np.max(group, axis=0)

    return update

def update_state(state: np.ndarray, update: np.ndarray, decay: float = 0.5) -> np.ndarray:
    """
    Update the state by applying a weighted decay with the given update values.

    Args:
        state (np.ndarray): A 2D array of shape (64, 12) representing the current state.
        update (np.ndarray): A 2D array of shape (64, 12) representing the updates to apply.
        decay (float): The decay factor to apply to the old state (default is 0.5).

    Returns:
        np.ndarray: The updated state.
    """
    for i in range(64):
        for j in range(12):
            state[i][j] = decay * state[i][j] + (1 - decay) * update[i][j]
    return state

def san_to_lan(board, san: str) -> str:
    """
    Converts a SAN move string to a LAN format by pushing the move to the board and then converting it.

    Args:
        board: A chess board object (e.g., from python-chess).
        san (str): A move in SAN (Standard Algebraic Notation) format.

    Returns:
        str: The corresponding move in LAN (Long Algebraic Notation) format.
    """
    board.push_san(san)
    history = board.move_stack
    lan = history[-1].uci()
    board.pop()
    return lan
