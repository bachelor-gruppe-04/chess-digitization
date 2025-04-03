import tensorflow as tf
import chess
import numpy as np
import asyncio

from typing import List, Tuple
from constants import SQUARE_NAMES, PIECE_SYMBOLS


from detection_methods import extract_xy_from_corners_mapping
from warp import get_inv_transform, transform_centers, transform_boundary
from map_pieces import detect, get_squares, get_update

async def find_fen(pieces_model_ref, frame, board_corners):
    keypoints = extract_xy_from_corners_mapping(board_corners, frame)
    inv_transform = get_inv_transform(keypoints)
    centers, centers3D = transform_centers(inv_transform)
    boundary, boundary3D = transform_boundary(inv_transform)
    
    print("insie fen")
    
    boxes, scores = await detect(frame, pieces_model_ref, keypoints)
    squares = get_squares(boxes, centers3D, boundary3D)
    state = get_update(scores, squares) 
    fen = set_fen_from_state(state)
    
    tf.keras.backend.clear_session()
    
    return fen   
    
def set_fen_from_state(state):
    assignment = [-1] * 64
    
    # Assign black king
    best_black_king_score = -1
    best_black_king_idx = -1
    for i in range(64):
        black_king_score = state[i][1]
        if black_king_score > best_black_king_score:
            best_black_king_score = black_king_score
            best_black_king_idx = i
    assignment[best_black_king_idx] = 1
    
    # Assign white king
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
    
    # Assign remaining pieces
    remaining_piece_idxs = [0, 2, 3, 4, 5, 6, 8, 9, 10, 11]
    square_names = [chess.square_name(i) for i in range(64)]
    
    for i in range(64):
        if assignment[i] != -1:
            continue
        
        best_idx = None
        best_score = 0.3
        
        for j in remaining_piece_idxs:
            square = square_names[i]
            bad_rank = square[1] in ('1', '8')
            is_pawn = PIECE_SYMBOLS[j % 6] == 'p'
            
            if is_pawn and bad_rank:
                continue
            
            score = state[i][j]
            if score > best_score:
                best_idx = j
                best_score = score
        
        if best_idx is not None:
            assignment[i] = best_idx
    
    # Set up the board
    board = chess.Board()
    board.clear()
    
    for i in range(64):
        if assignment[i] == -1:
            continue
        
        piece = PIECE_SYMBOLS[assignment[i] % 6]
        piece_color = chess.WHITE if assignment[i] > 5 else chess.BLACK
        square = chess.square(i % 8, i // 8)
        
        board.set_piece_at(square, chess.Piece.from_symbol(piece.upper() if piece_color == chess.WHITE else piece))
        
    return board.fen()


def get_fen_and_error(board: chess.Board, color: str):
    fen = board.fen()
    other_color = 'b' if color == 'w' else 'w'
    fen = fen.replace(f" {other_color} ", f" {color} ")

    error = None

    # Side to move has opponent in check
    for i in range(64):
        square = chess.SQUARE_NAMES[i]
        piece = board.piece_at(square)

        if piece is None:
            continue

        is_king = piece.piece_type == chess.KING
        is_other_color = piece.color == (chess.WHITE if other_color == 'w' else chess.BLACK)
        is_attacked = board.is_attacked_by(chess.WHITE if color == 'w' else chess.BLACK, square)

        if is_king and is_other_color and is_attacked:
            error = "Side to move has opponent in check"
            return fen, error

    return fen, error


