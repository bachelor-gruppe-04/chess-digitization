import chess
from constants import START_FEN
import chess
import chess.pgn
import io

class Game:
    def __init__(self, game_id: str, fen: str = START_FEN, moves: str = "", start: str = START_FEN, last_move: str = "", greedy: bool = False):
        self.id = game_id  # Unique game identifier
        self.fen = fen  # FEN representation of the board
        self.moves = moves  # PGN-like move history
        self.start = start  # Initial FEN or start position
        self.last_move = last_move  # Last move played
        self.greedy = greedy  # Boolean flag for greedy mode
        self.board = chess.Board(fen)  # Chess board initialized from FEN

    def update_last_move(self, move: str):
        """Updates last move, adds it to history, and updates the board."""
        self.last_move = move
        self.moves += f" {move}"  # Append move to PGN history
        self.board.push_san(move)  # Apply move to board

    def get_moves_pairs(self):
        """Returns a list of moves played."""
        return self.moves.strip().split() if self.moves else []

    def get_fen(self):
        """Returns the current board state in FEN notation."""
        return self.board.fen()

    def make_pgn(self):
        """Generates a PGN string including the starting FEN."""
        return f'[FEN "{self.start}"]\n\n{self.moves}'

    def reset_game(self):
        """Resets the game to its initial state."""
        self.fen = self.start
        self.moves = ""
        self.last_move = ""
        self.board = chess.Board(self.start)
        
        
    

def get_moves_from_pgn(board):
    pgn = board.board_fen()

    # Get rid of the headers (strings beginning with "[" and ending with "]")
    # Get rid of newline characters
    moves = ''.join([line for line in pgn.splitlines() if not line.startswith("[")])
    return moves.replace("\r", "").replace("\n", "")
def make_update_payload(board: chess.Board, greedy: bool = False):
    # Get the history of moves
    history = board.move_stack  # This gives the history of moves played so far

    # Get the moves as PGN (you can keep this if you need to)
    moves = get_moves_from_pgn(board)
    
    # Get the FEN string of the current position
    fen = board.fen()
    
    last_move = "" if len(history) == 0 else history[-1].uci()  # Using UCI for the last move (e.g., "e2e4")

    # Directly construct the payload
    payload = {
        "moves": moves,
        "fen": fen,
        "greedy": greedy,
        "lastMove": last_move
    }

    return payload
