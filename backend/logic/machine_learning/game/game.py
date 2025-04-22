import chess
import chess.pgn

from utilities.constants import START_FEN


class Game:
    def __init__(self, game_id: str, fen: str = START_FEN, moves: str = "", start: str = START_FEN, last_move: str = "", greedy: bool = False):
        self.id: str = game_id  # Unique game identifier
        self.fen: str = fen  # FEN representation of the board
        self.moves: str = moves  # PGN-like move history
        self.start: str = start  # Initial FEN or start position
        self.last_move: str = last_move  # Last move played
        self.greedy: bool = greedy  # Boolean flag for greedy mode
        self.board: chess.Board = chess.Board(fen)  # Chess board initialized from FEN

    def get_moves_pairs(self) -> list[str]:
        """Returns a list of moves played."""
        return self.moves.strip().split() if self.moves else []

    def get_fen(self) -> str:
        """Returns the current board state in FEN notation."""
        return self.board.fen()

def get_moves_from_pgn(board: chess.Board) -> str:
    """Convert board history to PGN string."""
    game = chess.pgn.Game.from_board(board)

    # Create a PGN exporter
    exporter = chess.pgn.StringExporter(headers=False, variations=False, comments=False)
    
    pgn = game.accept(exporter)
    return pgn.replace("\n", " ").replace("\r", "")

def make_update_payload(board: chess.Board, greedy: bool = False) -> dict:
    """Generate the payload containing board state, moves, and other details."""
    moves = get_moves_from_pgn(board)  # Assuming you have this function
    fen = board.fen()
    last_move = board.peek().uci() if board.move_stack else ""

    payload: dict = {
        "moves": moves,
        "fen": fen,
        "lastMove": last_move,
        "greedy": greedy
    }

    return payload