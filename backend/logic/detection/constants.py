import chess

START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MODEL_WIDTH = 480
MODEL_HEIGHT = 288
MARKER_RADIUS = 25
MARKER_DIAMETER = 2 * MARKER_RADIUS
CORNER_KEYS = ["h1", "a1", "a8", "h8"]
SQUARE_SIZE = 128
BOARD_SIZE = 8 * SQUARE_SIZE
LABELS = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]
PIECE_SYMBOLS = ["b", "k", "n", "p", "q", "r"]
PALETTE = [
    "#FF3838",  # Red
    "#FF9D97",  # Light red
    "#FF701F",  # Orange
    "#FFB21D",  # Light orange
    "#CFD231",  # Yellow
    "#48F90A",  # Green
    "#92CC17",  # Light green
    "#3DDB86",  # Light green
    "#1A9334",  # Dark green
    "#00D4BB",  # Cyan
    "#2C99A8",  # Blue
    "#00C2FF",  # Sky blue
]

SQUARE_NAMES = [
    'a1', 'b1', 'c1', 'd1', 'e1', 'f1', 'g1', 'h1',
    'a2', 'b2', 'c2', 'd2', 'e2', 'f2', 'g2', 'h2',
    'a3', 'b3', 'c3', 'd3', 'e3', 'f3', 'g3', 'h3',
    'a4', 'b4', 'c4', 'd4', 'e4', 'f4', 'g4', 'h4',
    'a5', 'b5', 'c5', 'd5', 'e5', 'f5', 'g5', 'h5',
    'a6', 'b6', 'c6', 'd6', 'e6', 'f6', 'g6', 'h6',
    'a7', 'b7', 'c7', 'd7', 'e7', 'f7', 'g7', 'h7',
    'a8', 'b8', 'c8', 'd8', 'e8', 'f8', 'g8', 'h8'
]

def make_square_map():
    return {square: i for i, square in enumerate(SQUARE_NAMES)}

SQUARE_MAP = make_square_map()



def make_label_map():
    return {label: i for i, label in enumerate(LABELS)}

LABEL_MAP = make_label_map()

