import chess
import chess.pgn


def save_game(board):
    game = chess.pgn.Game()
    game.setup(chess.Board())
    game.add_line(board.move_stack)

    # Save the PGN to a file
    with open("game.pgn", "w") as f:
        f.write(str(game))    