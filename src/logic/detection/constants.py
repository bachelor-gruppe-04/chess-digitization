START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
MODEL_WIDTH = 480
MODEL_HEIGHT = 288
MARKER_RADIUS = 25
MARKER_DIAMETER = 2 * MARKER_RADIUS
CORNER_KEYS = ["h1", "a1", "a8", "h8"]
SQUARE_SIZE = 128
BOARD_SIZE = 8 * SQUARE_SIZE
PALETTE = [
    "#FF3838", "#FF9D97", "#FF701F", "#FFB21D", "#CFD231",
    "#48F90A", "#92CC17", "#3DDB86", "#1A9334", "#00D4BB",
    "#2C99A8", "#00C2FF"
]

# Mapping of named corner keys to list indices
# CORNER_MAPPING = {
#     "h1": 0,  # First corner in the list
#     "a1": 1,  # Second corner in the list
#     "a8": 2,  # Third corner in the list
#     "h8": 3   # Fourth corner in the list
# }

LABELS = ["b", "k", "n", "p", "q", "r", "B", "K", "N", "P", "Q", "R"]
