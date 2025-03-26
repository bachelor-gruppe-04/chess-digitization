import asyncio
import numpy as np
import tensorflow as tf
import cv2

from detection_methods import get_input, get_boxes_and_scores
from piece_detection import predict_pieces


def find_pieces(modelPiecesRef, video, board_corners, board,
                 moves_pairs, last_move, mode):
    centers = None
    boundary = None
    centers3D = None
    boundary3D = None
    state = None
    keypoints = None
    possible_moves = set()
    greedy_move_to_time = {}
    
    
    async def loop():
        nonlocal centers, boundary, centers3D, boundary3D, state, keypoints, possible_moves, greedy_move_to_time
        
        while True:
                if centers is None:
                    state = np.zeros((64, 12))
                    possible_moves = set()
                    greedy_move_to_time = {}
                
                start_time = cv2.getTickCount()
                start_tensors = len(tf.config.experimental.list_physical_devices('GPU'))
                
                boxes, scores = await detect(modelPiecesRef, video, keypoints)
                squares = get_squares(boxes, centers3D, boundary3D)
                update = get_update(scores, squares)
                state = update_state(state, update)
                
                best_score1, best_score2, best_joint_score, best_move, best_moves = \
                    process_state(state, moves_pairs[0], possible_moves)
                
                end_time = cv2.getTickCount()
                fps = cv2.getTickFrequency() / (end_time - start_time)
                
                has_move = False
                if best_moves and mode != "play":
                    move = best_moves.sans[0]
                    has_move = best_score2 > 0 and best_joint_score > 0 and move in possible_moves
                    if has_move:
                        board.move(move)
                        possible_moves.clear()
                        greedy_move_to_time = {}
                
                has_greedy_move = False
                if best_move and not has_move and best_score1 > 0:
                    move = best_move.sans[0]
                    if move not in greedy_move_to_time:
                        greedy_move_to_time[move] = end_time
                    
                    second_elapsed = (end_time - greedy_move_to_time[move]) / cv2.getTickFrequency() > 1
                    new_move = san_to_lan(board, move) != last_move[0]
                    has_greedy_move = second_elapsed and new_move
                    
                    if has_greedy_move:
                        board.move(move)
                        greedy_move_to_time = {move: greedy_move_to_time[move]}
                
                if has_move or has_greedy_move:
                    greedy = False if mode == "play" else has_greedy_move
                    payload = make_update_payload(board, greedy)
                    print("payload", payload)
                    dispatch(game_update(payload))
                
                render_state(canvas, centers, boundary, state)
                
                del boxes, scores  # TensorFlow garbage collection
                
                end_tensors = len(tf.config.experimental.list_physical_devices('GPU'))
                if start_tensors < end_tensors:
                    print(f"Memory Leak! ({end_tensors} > {start_tensors})")
            
                await asyncio.sleep(0)
        
    asyncio.run(loop())




def calculate_score(state, move, from_thr=0.6, to_thr=0.6):
    score = 0
    for square in move['from']:
        score += 1 - max(state[square]) - from_thr
    
    for i in range(len(move['to'])):
        score += state[move['to'][i]][move['targets'][i]] - to_thr
    
    return score

def process_state(state, moves_pairs, possible_moves):
    best_score1 = float('-inf')
    best_score2 = float('-inf')
    best_joint_score = float('-inf')
    best_move = None
    best_moves = None
    seen = set()
    
    for move_pair in moves_pairs:
        if move_pair['move1']['sans'][0] not in seen:
            seen.add(move_pair['move1']['sans'][0])
            score = calculate_score(state, move_pair['move1'])
            
            if score > 0:
                possible_moves.add(move_pair['move1']['sans'][0])
            
            if score > best_score1:
                best_move = move_pair['move1']
                best_score1 = score
        
        if move_pair['move2'] is None or move_pair['moves'] is None or move_pair['move1']['sans'][0] not in possible_moves:
            continue
        
        score2 = calculate_score(state, move_pair['move2'])
        if score2 < 0:
            continue
        if score2 > best_score2:
            best_score2 = score2
        
        joint_score = calculate_score(state, move_pair['moves'])
        if joint_score > best_joint_score:
            best_joint_score = joint_score
            best_moves = move_pair['moves']
    
    return best_score1, best_score2, best_joint_score, best_move, best_moves


def get_box_centers(boxes):
    # Using tf.GradientTape or tf.function isn't required here since we're just performing tensor operations.
    with tf.GradientTape(persistent=True) as tape:
        # Slice the boxes tensor to get l, r, and b.
        l = boxes[:, 0:1]
        r = boxes[:, 2:3]
        b = boxes[:, 3:4]

        # Calculate the center coordinates.
        cx = (l + r) / 2
        cy = b - (r - l) / 3

        # Concatenate cx and cy to get the box centers.
        box_centers = tf.concat([cx, cy], axis=1)

    return box_centers


def get_squares(boxes, centers3D, boundary3D):
    with tf.GradientTape(persistent=True) as tape:
        # Expand dims for box centers (equivalent to tf.expandDims)
        box_centers_3d = tf.expand_dims(get_box_centers(boxes), axis=1)
        print(box_centers_3d)
        print(centers3D)

        # Calculate squared distance (equivalent to tf.sub, tf.square, tf.sum)
        dist = tf.reduce_sum(tf.square(box_centers_3d - centers3D), axis=2)
        squares = tf.argmin(dist, axis=1)

        # Adjust boundary (equivalent to tf.concat and tf.slice)
        shifted_boundary_3d = tf.concat([
            tf.slice(boundary3D, [0, 1, 0], [1, 3, 2]),
            tf.slice(boundary3D, [0, 0, 0], [1, 1, 2])
        ], axis=1)

        print(boundary3D)
        print(shifted_boundary_3d)

        # Get the number of boxes (equivalent to tf.shape)
        n_boxes = box_centers_3d.shape[0]

        a = tf.squeeze(tf.subtract(
            tf.cast(tf.slice(boundary3D, [0, 0, 0], [1, 4, 1]), tf.float32),
            tf.cast(tf.slice(shifted_boundary_3d, [0, 0, 0], [1, 4, 1]), tf.float32)
        ), axis=2)

        b = tf.squeeze(tf.subtract(
            tf.cast(tf.slice(boundary3D, [0, 0, 1], [1, 4, 1]), tf.float32),
            tf.cast(tf.slice(shifted_boundary_3d, [0, 0, 1], [1, 4, 1]), tf.float32)
        ), axis=2)

        c = tf.squeeze(tf.subtract(
            tf.cast(tf.slice(box_centers_3d, [0, 0, 0], [n_boxes, 1, 1]), tf.float32),
            tf.cast(tf.slice(shifted_boundary_3d, [0, 0, 0], [1, 4, 1]), tf.float32)
        ), axis=2)

        d = tf.squeeze(tf.subtract(
            tf.cast(tf.slice(box_centers_3d, [0, 0, 1], [n_boxes, 1, 1]), tf.float32),
            tf.cast(tf.slice(shifted_boundary_3d, [0, 0, 1], [1, 4, 1]), tf.float32)
        ), axis=2)

        # Calculate determinant (equivalent to tf.mul and tf.sub)
        det = tf.subtract(tf.multiply(a, d), tf.multiply(b, c))

        # Apply condition (equivalent to tf.where, tf.any, tf.less)
        new_squares = tf.where(
        tf.reduce_any(tf.less(det, 0), axis=1),
        tf.cast(tf.fill(tf.shape(squares), -1), tf.int32),  # Cast to int32
        tf.cast(squares, tf.int32)  # Cast squares to int32
    )

        return new_squares.numpy()  # To convert the tensor back to a numpy array

    return squares

def get_update(scores_tensor, squares):
    update = np.zeros((64, 12))
    scores = scores_tensor

    for i in range(len(squares)):
        square = squares[i]
        if square == -1:
            continue
        for j in range(12):
            update[square][j] = max(update[square][j], scores[i][j])

    return update

def update_state(state, update, decay=0.5):
    for i in range(64):
        for j in range(12):
            state[i][j] = decay * state[i][j] + (1 - decay) * update[i][j]
    return state


def san_to_lan(board, san):
    board.push_san(san)
    history = board.move_stack
    lan = history[-1].uci()
    board.pop()
    return lan
  
  
async def detect(frame, pieces_model_ref, keypoints):
    frame_height, frame_width, _ = frame.shape

    image4d, width, height, padding, roi = get_input(frame, keypoints)

    pieces_prediction = predict_pieces(frame, pieces_model_ref)
    boxes, scores = get_boxes_and_scores(pieces_prediction, width, height, frame_width, frame_height, padding, roi)

    del pieces_prediction
    del image4d  

    return boxes, scores

