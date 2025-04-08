import asyncio
import numpy as np
import tensorflow as tf
import cv2
import time

from detection_methods import get_input, get_boxes_and_scores
from piece_detection import predict_pieces
from game import make_update_payload

from detection_methods import extract_xy_from_corners_mapping
from warp import get_inv_transform, transform_centers, transform_boundary


async def find_pieces(piece_model_ref, video_ref, corners_ref, game_ref, moves_pairs_ref):
    centers = None
    boundary = None
    centers_3d = None
    boundary_3d = None
    state = None
    keypoints = None
    possible_moves = set()
    greedy_move_to_time = {}
    
    print("piece_model_ref")    
    print(piece_model_ref)


    async def loop():
        nonlocal centers, boundary, centers_3d, boundary_3d, state, keypoints, possible_moves, greedy_move_to_time
        if False:
            centers = None
        else:
            if centers is None:
                keypoints = extract_xy_from_corners_mapping(corners_ref, video_ref)
                inv_transform = get_inv_transform(keypoints)
                centers, centers_3d = transform_centers(inv_transform)
                boundary, boundary_3d = transform_boundary(inv_transform)
                state = np.zeros((64, 12))
                possible_moves = set()
                greedy_move_to_time = {}

            start_time = time.time()
            

            boxes, scores = await detect(piece_model_ref, video_ref, keypoints)
            

            # Now updated_boxes is a TensorFlow tensor with the modified values
            squares = get_squares(boxes, centers_3d, boundary_3d)
            
            update = get_update(scores, squares)
            # np.set_printoptions(threshold=np.inf)
            colors = [
        (255, 0, 0),        # Blue
        (0, 255, 0),        # Green
        (0, 0, 255),        # Red
        (0, 255, 255),      # Yellow
        (255, 0, 255),      # Magenta
        (255, 255, 0),      # Cyan
        (0, 0, 0),          # Black
        (255, 255, 255)     # White
    ]   
            
            
            # box_centers_np = tf.squeeze(box_centers_3D).numpy()  # shape [n, 2]

            # for (x, y) in box_centers_np:
            #     x_int, y_int = int(x), int(y)
                # cv2.circle(video_ref, (x_int, y_int), radius=5, color=(0, 0, 255), thickness=-1)  # red dot
                        
                    # visualize box rectangles
                # Assuming 'boxes' contains the bounding boxes and 'colors' is the list of colors you want to use

            for i, box in enumerate(boxes.numpy()):
                # Only display every 20th box
                if i % 33 == 0:
                    x1, y1, x2, y2 = map(int, box)
                    
                    # Assign a color from the list based on the index
                    color = colors[i % len(colors)]  # This ensures that if there are more boxes than colors, it will cycle through the colors
                    
                    # Draw the rectangle with the selected color
                    cv2.rectangle(video_ref, (x1, y1), (x2, y2), color, 2)
                    
                    # Annotate the box with its index
                    cv2.putText(video_ref, f"Box{i}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


            # Visualize centers
            for i, (x, y) in enumerate(centers):
                cv2.circle(video_ref, (int(x), int(y)), 5, (0, 255, 0), -1)
                cv2.putText(video_ref, f"C{i}", (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            # Visualize boundary
            boundary_np = np.array(boundary, dtype=np.int32)
            cv2.polylines(video_ref, [boundary_np], isClosed=True, color=(255, 0, 0), thickness=2)
            for i, (x, y) in enumerate(boundary):
                cv2.circle(video_ref, (int(x), int(y)), 4, (255, 0, 0), -1)
                cv2.putText(video_ref, f"B{i}", (int(x) + 5, int(y) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

            cv2.imshow("Detected Pieces", video_ref)
            cv2.waitKey(1)


            
            state = update_state(state, update)
        
        
            best_score1, best_score2, best_joint_score, best_move, best_moves = process_state(state, moves_pairs_ref, possible_moves)
            print("best move and best moves")
            print(best_move)
            print(best_moves)
            print("best score")
            print(best_score1)
            print(best_score2)
            print(best_joint_score)
            print("possible moves")
            print(possible_moves)
            end_time = time.time()
            fps = round(1 / (end_time - start_time), 1)
            
            print("FPS")
            print(fps)

            has_move = False
            if best_moves is not None:
                move = best_moves["sans"][0]
                has_move = best_score2 > 0 and best_joint_score > 0 and move in possible_moves
                if has_move:
                    game_ref.board.push_san(move)
                    possible_moves.clear()
                    greedy_move_to_time = {}

            has_greedy_move = False
            if best_move is not None and not has_move and best_score1 > 0:
                move = best_move["sans"][0]
                if move not in greedy_move_to_time:
                    greedy_move_to_time[move] = end_time

                second_elapsed = (end_time - greedy_move_to_time[move]) > 1  # 1000 ms = 1 second
                new_move = san_to_lan(game_ref.board, move) != game_ref["last_move"]
                has_greedy_move = second_elapsed and new_move
                if has_greedy_move:
                    game_ref["board"].move(move)
                    greedy_move_to_time = {move: greedy_move_to_time[move]}

            if has_move or has_greedy_move:
                greedy = False
                payload = make_update_payload(game_ref.board, greedy)
                print("payload")
                print(payload)
                # dispatch(game_update(payload))
                
                

            # Dispose of the tensors to free memory
            tf.keras.backend.clear_session()

        await loop()  # Correctly awaiting the recursive call

    # Initial call to start the loop 
    await loop()

    # Clean up when the function is called to terminate
    def cleanup():
        tf.keras.backend.clear_session()
        # Implement cancellation if necessary (e.g., clearing loops or canceling animations)
    
    return cleanup



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
        
        print("det")
        np.set_printoptions(threshold=np.inf)
        print(det)
        np.set_printoptions(threshold=1)

        
        np.set_printoptions(threshold=1)

        print("types")
        print("box_centers_3D")
        print(box_centers_3D)
        print(type(box_centers_3D))
        print("centers3d")
        print(centers3D)
        print(type(centers3D))
        print("boundary3D")
        print(boundary3D)
        print(type(boundary3D))
        
        print("squares")
        print(squares)
        print(type(squares))

        # Apply tf.where condition for negative det values
        new_squares = tf.where(
            tf.reduce_any(tf.less(det, 0), axis=1),  # Check if any det < 0 along axis 1
            tf.constant(-1, dtype=squares.dtype),    # Replace with -1
            squares                                   # Otherwise, keep original squares
        )
        
        print("newsquares")
        np.set_printoptions(threshold=np.inf)
        print(new_squares)
        np.set_printoptions(threshold=1)
        
        
        return squares

def get_update(scores_tensor, squares):
    update = np.zeros((64, 12))
    scores = scores_tensor.numpy()
    
    print(squares)
    print("squares")
    print(scores_tensor)
    print("scores")

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
  
  
async def detect(pieces_model_ref, video_ref, keypoints):
    frame_height, frame_width, _ = video_ref.shape

    image4d, width, height, padding, roi = get_input(video_ref, keypoints)

    pieces_prediction = predict_pieces(image4d, pieces_model_ref)
    boxes, scores = get_boxes_and_scores(pieces_prediction, width, height, frame_width, frame_height, padding, roi)
    
    print("inside detect")
    
    
    # print(boxes)
    
    np.set_printoptions(threshold=5, linewidth=100, edgeitems=10)

    print(scores)
    
    
    np.set_printoptions()  # Resets everything to default

    

    del pieces_prediction
    del image4d  

    return boxes, scores