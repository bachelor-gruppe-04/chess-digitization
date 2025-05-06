# import tensorflow as tf
# import numpy as np
# import chess

# from typing import List, Tuple, Optional
# from logic.machine_learning.utilities.constants import PIECE_SYMBOLS
# from logic.machine_learning.maths.warp import get_inv_transform, transform_centers, transform_boundary
# from logic.machine_learning.board_statestate.map_pieces import detect, get_squares, get_update
# from logic.machine_learning.detection.corners_detection import extract_xy_from_labeled_corners

# async def find_fen(pieces_model_ref: tf.Module, frame: np.ndarray, board_corners: np.ndarray) -> str:
#     """
#     This function processes the given frame to detect the board state, and then generates the FEN 
#     (Forsyth-Edwards Notation) string representing the current state of the chessboard.

#     Args:
#         pieces_model_ref (tf.Module): A TensorFlow model used for piece detection.
#         frame (np.ndarray): The image frame containing the board.
#         board_corners (np.ndarray): Coordinates of the corners of the chessboard in the frame.

#     Returns:
#         str: The FEN string representing the current state of the chessboard.
#     """
#     # Extract the keypoints and transform them
#     keypoints = extract_xy_from_labeled_corners(board_corners, frame)
#     inv_transform = get_inv_transform(keypoints)
#     centers, centers3D = transform_centers(inv_transform)
#     boundary, boundary3D = transform_boundary(inv_transform)
    
#     # Perform detection to get bounding boxes and scores
#     boxes, scores = await detect(pieces_model_ref, frame, keypoints)
#     del pieces_model_ref  # Clear the model reference after detection
    
#     # Get squares and state from the detection data
#     squares = get_squares(boxes, centers3D, boundary3D)
#     state = get_update(scores, squares) 
    
#     # Generate and return the FEN string
#     fen = set_fen_from_state(state)
#     return fen  


# def set_fen_from_state(state: np.ndarray) -> str:
#     """
#     Converts the state (a 2D array) into a FEN string representing the chessboard position.
    
#     Args:
#         state (np.ndarray): A 2D numpy array where each element represents a piece's score at a given position.

#     Returns:
#         str: The FEN string representing the board state.
#     """
#     assignment = [-1] * 64  # Initialize the assignment with -1 (no piece assigned)

#     # Assign black king
#     best_black_king_score = -1
#     best_black_king_idx = -1
#     for i in range(64):
#         black_king_score = state[i][1]
#         if black_king_score > best_black_king_score:
#             best_black_king_score = black_king_score
#             best_black_king_idx = i
#     assignment[best_black_king_idx] = 1  # Assign the black king
    
#     # Assign white king
#     best_white_king_score = -1
#     best_white_king_idx = -1
#     for i in range(64):
#         if i == best_black_king_idx:
#             continue
#         white_king_score = state[i][7]
#         if white_king_score > best_white_king_score:
#             best_white_king_score = white_king_score
#             best_white_king_idx = i
#     assignment[best_white_king_idx] = 7  # Assign the white king
    
#     # Assign remaining pieces
#     remaining_piece_idxs = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]
#     square_names = [chess.square_name(i) for i in range(64)]
    
#     for i in range(64):
#         if assignment[i] != -1:
#             continue  # Skip already assigned squares
        
#         best_idx = None
#         best_score = 0.3  # A threshold for piece detection
        
#         for j in remaining_piece_idxs:
#             square = square_names[i]
#             bad_rank = square[1] in ('1', '8')  # Bad ranks for pawns
#             is_pawn = PIECE_SYMBOLS[j % 6] == 'p'
            
#             # If the piece is a pawn and it's in the first or last rank, skip it
#             if is_pawn and bad_rank:
#                 continue
            
#             score = state[i][j]
#             if score > best_score:
#                 best_idx = j
#                 best_score = score
        
#         if best_idx is not None:
#             assignment[i] = best_idx  # Assign piece to this square
    
#     # Set up the board using chess library
#     board = chess.Board()
#     board.clear()
    
#     # Place the pieces on the board according to the assignment
#     for i in range(64):
#         if assignment[i] == -1:
#             continue
        
#         piece = PIECE_SYMBOLS[assignment[i] % 6]
#         piece_color = chess.WHITE if assignment[i] > 5 else chess.BLACK
#         square = chess.square(i % 8, i // 8)  # Convert index to square
        
#         # Set the piece at the square on the board
#         board.set_piece_at(square, chess.Piece.from_symbol(piece.upper() if piece_color == chess.WHITE else piece))
    
#     # Return the FEN string representing the board
#     return board.fen()
