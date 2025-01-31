import chess
import chess.pgn
import chess_analysis

board = chess.Board()

def random_moves(board):
    board.push_san('e4')
    board.push_san('e5')
    board.push_san('Nf3')
    board.push_san('Nc6')


def save_game(board):
    game = chess.pgn.Game()
    game.setup(chess.Board())
    game.add_line(board.move_stack)

    # Save the PGN to a file
    with open("game.pgn", "w") as f:
        f.write(str(game))    


random_moves(board)
save_game(board)
chess_analysis.analyze_position(board.fen())
print(board.fen())
