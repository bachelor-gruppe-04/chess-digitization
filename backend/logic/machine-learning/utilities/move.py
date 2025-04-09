import chess

from utilities.constants import SQUARE_MAP, LABEL_MAP


# Castling map to handle rook movement during castling
# Maps king's destination square to: (rook start square, rook end square, rook label)
castling_map = {
    "g1": [SQUARE_MAP["h1"], SQUARE_MAP["f1"], LABEL_MAP["R"]],
    "c1": [SQUARE_MAP["a1"], SQUARE_MAP["d1"], LABEL_MAP["R"]],
    "g8": [SQUARE_MAP["h8"], SQUARE_MAP["f8"], LABEL_MAP["r"]],
    "c8": [SQUARE_MAP["a8"], SQUARE_MAP["d8"], LABEL_MAP["r"]]
}

def get_piece_idx(board, move):
    """
    Returns the label index of the piece involved in a move.

    Args:
        board (chess.Board): The current board state.
        move (chess.Move): The move being analyzed.

    Returns:
        int or None: The index from LABEL_MAP corresponding to the piece symbol,
                     or None if no piece was found at the source square.
    """
    piece = board.piece_at(move.from_square)
    if not piece:
        return None

    piece_symbol = piece.symbol()

    # Handle promotion case
    if move.promotion:
        piece_symbol = chess.Piece(move.promotion, piece.color).symbol()

    return LABEL_MAP.get(piece_symbol)


def get_data(board, move):
    """
    Gathers data for a single move, including from/to squares and piece identity.

    Handles special moves such as castling and en passant.

    Args:
        board (chess.Board): The board state before the move.
        move (chess.Move): The move to be analyzed.

    Returns:
        dict: A dictionary with:
            - 'sans': [move in standard algebraic notation],
            - 'from': list of squares the pieces move from,
            - 'to': list of destination squares,
            - 'targets': list of piece labels involved in the move.
    """
    from_square_idx = move.from_square
    to_square_idx = move.to_square

    from_squares = [from_square_idx]
    to_squares = [to_square_idx]
    targets = [get_piece_idx(board, move)]

    if board.is_castling(move):
        # Handle castling moves (kingside or queenside)
        rook_from, rook_to = castling_map.get(move.to_square, (None, None))
        if rook_from and rook_to:
            from_squares.append(rook_from)
            to_squares.append(rook_to)
            targets.append("rook")

    elif board.is_en_passant(move):
        # Handle en-passant capture
        captured_pawn_square = chess.square_name(chess.square(
            chess.square_file(move.to_square), chess.square_rank(move.from_square)
        ))
        from_squares.append(captured_pawn_square)

    move_data = {
        "sans": [board.san(move)],  # Standard algebraic notation
        "from": from_squares,
        "to": to_squares,
        "targets": targets
    }
    return move_data



def combine_data(move1_data, move2_data):
    """
    Combines data from two sequential moves, avoiding overlapping squares.

    Args:
        move1_data (dict): Output of get_data for the first move.
        move2_data (dict): Output of get_data for the second move.

    Returns:
        dict: A merged move data dictionary with cleaned `from`, `to`, and `targets` lists.
    """
    bad_squares = move2_data["from"] + move2_data["to"]
    from1 = [x for x in move1_data["from"] if x not in bad_squares]
    
    to1 = []
    targets1 = []
    for i in range(len(move1_data["to"])):
        if move1_data["to"][i] in bad_squares:
            continue
        to1.append(move1_data["to"][i])
        targets1.append(move1_data["targets"][i])
    
    from_combined = from1 + move2_data["from"]
    to_combined = to1 + move2_data["to"]
    targets_combined = targets1 + move2_data["targets"]
    
    data = {
        "sans": [move1_data["sans"][0], move2_data["sans"][0]],
        "from": from_combined,
        "to": to_combined,
        "targets": targets_combined
    }
    
    return data


def get_moves_pairs(board: chess.Board):
    """
    Generates all legal move pairs from the current board position.

    For each legal move (move1), all legal responses (move2) are computed
    and returned along with combined move data.

    Args:
        board (chess.Board): The current chess position.

    Returns:
        list: List of dictionaries with structure:
            - 'move1': data from the first move
            - 'move2': data from the second move (or None if no legal replies)
            - 'moves': combined move data (or None if no legal replies)
    """
    moves_pairs = []
    
    for move1 in list(board.legal_moves):  
        move1_data = get_data(board, move1)  
        board.push(move1)  # Make the first move
        done = True

        for move2 in list(board.legal_moves):  
            move2_data = get_data(board, move2)
            moves_data = combine_data(move1_data, move2_data)
            moves_pairs.append({
                "move1": move1_data,
                "move2": move2_data,
                "moves": moves_data
            })
            done = False  

        if done:  
            moves_pairs.append({
                "move1": move1_data,
                "move2": None,
                "moves": None
            })

        board.pop()  # Undo the first move
    
    return moves_pairs

