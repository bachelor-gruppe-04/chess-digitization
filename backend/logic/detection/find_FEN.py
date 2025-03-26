import tensorflow as tf
import chess

from chess import Board, Piece
from typing import List, Tuple
from constants import SQUARE_NAMES


from detection_methods import extract_xy_from_corners_mapping
from warp import get_inv_transform, transform_centers, transform_boundary
from map_pieces import detect, get_squares, get_update

async def find_fen(pieces_model_ref, frame, board_corners):
    keypoints = extract_xy_from_corners_mapping(board_corners, frame)
    inv_transform = get_inv_transform(keypoints)
    centers, centers3D = transform_centers(inv_transform)
    boundary, boundary3D = transform_boundary(inv_transform)
    
    boxes, scores = await detect(frame, pieces_model_ref, keypoints)
    print("boundary3d")
    print(boundary3D)
    squares = get_squares(boxes, centers3D, boundary3D)
    state = get_update(scores, squares)    
    set_fen_from_state(state)
    # render_state(frame, centers, boundary, state)
    
    tf.keras.backend.clear_session()
    
    
def set_fen_from_state(state: List[List[float]]):
    print("state")
    print(state)
    assignment = [-1] * 64

    # First pass: Assign the black king
    best_black_king_score = -1
    best_black_king_idx = -1
    for i in range(64):
        black_king_score = state[i][1]
        if black_king_score > best_black_king_score:
            best_black_king_score = black_king_score
            best_black_king_idx = i
    assignment[best_black_king_idx] = 1

    # Second pass: Assign the white king
    best_white_king_score = -1
    best_white_king_idx = -1
    for i in range(64):
        if i == best_black_king_idx:
            continue
        white_king_score = state[i][7]
        if white_king_score > best_white_king_score:
            best_white_king_score = white_king_score
            best_white_king_idx = i
    assignment[best_white_king_idx] = 7

    # Third pass: Assign the remaining pieces
    remaining_piece_idxs = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    piece_symbols = ["p", "n", "b", "r", "q", "k"]

    for i in range(64):
        if assignment[i] != -1:
            continue

        best_idx = None
        best_score = 0.3
        for j in remaining_piece_idxs:
            square = SQUARE_NAMES[i]
            bad_rank = square[1] in "18"
            is_pawn = piece_symbols[j % 6] == "p"
            
            if is_pawn and bad_rank:
                continue
            
            score = state[i][j]
            if score > best_score:
                best_idx = j
                best_score = score

        if best_idx is not None:
            assignment[i] = best_idx

    # Construct the board
    board = Board()
    board.clear()
    for i in range(64):
        if assignment[i] == -1:
            continue

        # Select the piece type from the symbols list
        piece_type = piece_symbols[assignment[i] % 6]

        # Determine the piece color ('w' for white, 'b' for black)
        piece_color = 'w' if assignment[i] > 5 else 'b'

        # Determine the square name using the square index (SQUARE_NAMES is assumed to be a list of square names)
        square = SQUARE_NAMES[i]

        # Create the piece symbol by combining the color and piece type
        # For white pieces, the symbol will be uppercase (e.g., 'N', 'K')
        # For black pieces, the symbol will be lowercase (e.g., 'n', 'k')
        piece_symbol = piece_type.upper() if piece_color == 'w' else piece_type.lower()
        
        
        square_index = chess.parse_square(square)

        board.set_piece_at(square_index, Piece.from_symbol(piece_symbol))

    # After all pieces have been set, get the FEN string
    fen = board.fen()
    print(fen)
    if board.is_valid():
        print("THIS IS FEN")
        print(fen)
        print(["Set starting FEN"])
    else:
        print(["Invalid FEN"])

    return fen

