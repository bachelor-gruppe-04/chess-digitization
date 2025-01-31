import chess
import chess.pgn
import requests

import requests

def get_stockfish_evaluation(fen, depth):
    url = "https://stockfish.online/api/s/v2.php" 

    params = {
        "fen": fen,
        "depth": depth
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        evaluation = response.json()
        return evaluation

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Stockfish API: {e}")
        return None

# Create a new chess board and make moves (your existing code)
board = chess.Board()
board.push_san('e4')
board.push_san('e5')
board.push_san('Nf3')
board.push_san('Nc6')

# Get the FEN of the board
fen = board.fen()
print(f"Current FEN: {fen}")

evaluation_result = get_stockfish_evaluation(fen, depth=15)  # You can adjust the depth


if evaluation_result:
    print(evaluation_result)
    if "evaluation" in evaluation_result:
        print(f"Stockfish evaluation: {evaluation_result['evaluation']}")

else:
    print("Failed to retrieve Stockfish evaluation.")