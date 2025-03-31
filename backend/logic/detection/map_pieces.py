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

def get_squares(boxes: tf.Tensor, centers3D: tf.Tensor, boundary3D: tf.Tensor):
    with tf.device('/CPU:0'):
        box_centers_3D = tf.expand_dims(get_box_centers(boxes), 1)
        dist = tf.reduce_sum(tf.square(box_centers_3D - centers3D), axis=2)
        squares = tf.argmin(dist, axis=1)
        
        shifted_boundary_3D = tf.concat([
            tf.slice(boundary3D, [0, 1, 0], [1, 3, 2]),
            tf.slice(boundary3D, [0, 0, 0], [1, 1, 2]),
        ], axis=1)
        
        n_boxes = tf.shape(box_centers_3D)[0]    
        
        print("n_boxes")
        print(n_boxes)
        print(box_centers_3D)
        print("boundary")
        print(boundary3D)
        print(shifted_boundary_3D)    
        
                
        a = tf.squeeze(tf.slice(boundary3D, [0, 0, 0], [1, 4, 1]) -
                       tf.slice(shifted_boundary_3D, [0, 0, 0], [1, 4, 1]), axis=[2])
        b = tf.squeeze(tf.slice(boundary3D, [0, 0, 1], [1, 4, 1]) -
                       tf.slice(shifted_boundary_3D, [0, 0, 1], [1, 4, 1]), axis=[2])
        c = tf.squeeze(tf.slice(box_centers_3D, [0, 0, 0], [n_boxes, 1, 1]) -
                       tf.slice(shifted_boundary_3D, [0, 0, 0], [1, 4, 1]), axis=[2])
        d = tf.squeeze(tf.slice(box_centers_3D, [0, 0, 1], [n_boxes, 1, 1]) -
                       tf.slice(shifted_boundary_3D, [0, 0, 1], [1, 4, 1]), axis=[2])

    
    
        det = tf.subtract(tf.multiply(a, d), tf.multiply(b, c))  # Equivalent to tf.sub and tf.mul in JS
        
        print("det")
        print(det)
        
        # np.set_printoptions(threshold=np.inf) 
        print("det values:", det.numpy())  # Ensure `det` has both positive and negative values
        print("det dtype:", det.dtype)
        print("det shape:", det.shape)
        # np.set_printoptions(threshold=np.inf) 
        print("Condition check:", tf.reduce_any(det < 0, axis=1).numpy())  # See which rows trigger `True`
        

        # Apply tf.where condition
        new_squares = tf.where(
            tf.reduce_any(tf.less(det, 0), axis=1),  # Check if any det < 0 along axis 1
            tf.constant(-1, dtype=squares.dtype),    # Replace with -1
            squares                                   # Otherwise, keep original squares
        )
        # return new_squares.numpy() 
        return squares

def get_update(scores_tensor, squares):
    update = np.zeros((64, 12))
    scores = scores_tensor.numpy()

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

