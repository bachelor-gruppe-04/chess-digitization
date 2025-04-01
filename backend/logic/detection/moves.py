import chess

from constants import SQUARE_MAP, LABEL_MAP

# Assuming SQUARE_MAP and LABEL_MAP are already defined somewhere

# Castling map
castling_map = {
    "g1": [SQUARE_MAP["h1"], SQUARE_MAP["f1"], LABEL_MAP["R"]],
    "c1": [SQUARE_MAP["a1"], SQUARE_MAP["d1"], LABEL_MAP["R"]],
    "g8": [SQUARE_MAP["h8"], SQUARE_MAP["f8"], LABEL_MAP["r"]],
    "c8": [SQUARE_MAP["a8"], SQUARE_MAP["d8"], LABEL_MAP["r"]]
}

# Get piece index
def get_piece_idx(move):
    piece = move.piece
    if move.promotion:
        piece = move.promotion
    if move.color == "w":
        piece = piece.upper()
    piece_idx = LABEL_MAP.get(piece)
    return piece_idx

# Get move data
def get_data(move):
    print("square map")
    print(SQUARE_MAP)
    print("move from square")
    print(move.from_square)
    print(SQUARE_MAP[move.from_square])
    from_squares = [SQUARE_MAP[move.from_square]]
    to_squares = [SQUARE_MAP[move.to_square]]
    targets = [get_piece_idx(move)]
    
    if "k" in move.flags or "q" in move.flags:
        # Castling
        from_sq, to_sq, target = castling_map[move.to_square]
        from_squares.append(from_sq)
        to_squares.append(to_sq)
        targets.append(target)
    elif "e" in move.flags:
        # En-passant
        captured_pawn_square = SQUARE_MAP[move.to_square[0] + move.from_square[1]]
        from_squares.append(captured_pawn_square)
    
    move_data = {
        "sans": [move.san],
        "from": from_squares,
        "to": to_squares,
        "targets": targets
    }
    return move_data

# Combine two moves data
def combine_data(move1_data, move2_data):
    bad_squares = move2_data["from"] + move2_data["to"]
    from1 = [x for x in move1_data["from"] if x not in bad_squares]

    to1 = []
    targets1 = []
    for i in range(len(move1_data["to"])):
        if move1_data["to"][i] not in bad_squares:
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

# Get all move pairs from a board
def get_moves_pairs(board):
    moves_pairs = []
    for move1 in board.legal_moves:
        move1_data = get_data(move1)
        board.push(move1)
        done = True
        
        for move2 in board.legal_moves:
            move2_data = get_data(move2)
            moves_data = combine_data(move1_data, move2_data)
            moves_pair = {
                "move1": move1_data,
                "move2": move2_data,
                "moves": moves_data
            }
            moves_pairs.append(moves_pair)
            done = False
        
        if done:
            moves_pair = {
                "move1": move1_data,
                "move2": None,
                "moves": None
            }
            moves_pairs.append(moves_pair)
            print(moves_pair)
        
        board.pop()  # Undo the move
    
    return moves_pairs
