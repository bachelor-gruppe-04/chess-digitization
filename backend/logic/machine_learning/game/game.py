import chess
import chess.pgn

from utilities.constants import START_FEN

class Game:
    def __init__(self, game_id: str, fen: str = START_FEN, moves: str = "", start: str = START_FEN, last_move: str = "", greedy: bool = False):
        self.id = game_id  # Unique game identifier
        self.fen = fen  # FEN representation of the board
        self.moves = moves  # PGN-like move history
        self.start = start  # Initial FEN or start position
        self.last_move = last_move  # Last move played
        self.greedy = greedy  # Boolean flag for greedy mode
        self.board = chess.Board(fen)  # Chess board initialized from FEN


    def get_moves_pairs(self):
        """Returns a list of moves played."""
        return self.moves.strip().split() if self.moves else []

    def get_fen(self):
        """Returns the current board state in FEN notation."""
        return self.board.fen()
        
    

def get_moves_from_pgn(board):
    # Convert board history to PGN
    game = chess.pgn.Game.from_board(board)

    # Create a PGN exporter
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    
    pgn = game.accept(exporter)
    return pgn.replace("\n", " ").replace("\r", "")


def make_update_payload(board: chess.Board, greedy: bool = False):
    moves = get_moves_from_pgn(board)  # Assuming you have this function
    fen = board.fen()
    last_move = board.peek().uci() if board.move_stack else ""

    payload = {
        "moves": moves,
        "fen": fen,
        "lastMove": last_move,
        "greedy": greedy
    }

    return payload
