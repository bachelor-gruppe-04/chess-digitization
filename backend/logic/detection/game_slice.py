import chess
import json
from typing import List, Dict, Optional
from constants import START_FEN

class Game:
    def __init__(self):
        self.moves = ""
        self.fen = START_FEN
        self.start = START_FEN
        self.last_move = ""
        self.greedy = False


def get_moves_from_pgn(board: chess.Board) -> str:
    # Get the PGN string of the game
    pgn = board.pgn()
    
    # Get rid of the headers (strings beginning with "[" and ending with "]")
    # Get rid of newline characters
    moves = pgn.replace(r'\[.*?\]', '').replace(r'\r?\n|\r', '')
    return moves


def make_pgn(game: Game) -> str:
    return f'[FEN "{game.start}"]\n\n{game.moves}'


def make_update_payload(board: chess.Board, greedy: bool = False) -> Dict[str, Optional[str]]:
    # Get the history of moves in verbose format
    history = board.move_stack  # This gives the history of moves played so far

    # Get the moves as PGN
    moves = get_moves_from_pgn(board)
    
    # Get the FEN string of the current position
    fen = board.fen()

    # Get the last move in LAN (algebraic notation)
    last_move = "" if len(history) == 0 else board.san(history[-1])

    # Construct the payload
    payload = {
        "moves": moves,
        "fen": fen,
        "lastMove": last_move,
        "greedy": greedy
    }

    return payload


def make_board(game: Game) -> chess.Board:
    # Create a new board and load the PGN from the game
    board = chess.Board(game.start)
    board.set_pgn(make_pgn(game))
    return board


# State management simulation
class GameSlice:
    def __init__(self):
        self.state = Game()

    def game_set_moves(self, moves: str):
        self.state.moves = moves

    def game_set_fen(self, fen: str):
        self.state.fen = fen

    def game_set_start(self, start: str):
        self.state.start = start

    def game_set_last_move(self, last_move: str):
        self.state.last_move = last_move

    def game_reset_moves(self):
        self.state.moves = ""

    def game_reset_fen(self):
        self.state.fen = START_FEN

    def game_reset_start(self):
        self.state.start = START_FEN

    def game_reset_last_move(self):
        self.state.last_move = ""

    def game_update(self, payload: Dict[str, Optional[str]]):
        self.state.moves = payload.get("moves", self.state.moves)
        self.state.fen = payload.get("fen", self.state.fen)
        self.state.last_move = payload.get("lastMove", self.state.last_move)
        self.state.greedy = payload.get("greedy", self.state.greedy)