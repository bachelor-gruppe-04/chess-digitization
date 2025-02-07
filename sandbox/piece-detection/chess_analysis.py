import requests
import chess

def stockfish_api(fen, depth):
    url = "https://chess-api.com/v1"

    payload = {
        "fen": fen,
        "depth": depth,
        "maxThinkingTime": 100, 
    }

    headers = {"Content-Type": "application/json"}  # Ensure JSON content-type

    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise exception for HTTP errors (4xx, 5xx)
        evaluation = response.json()
        return evaluation

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Chess API: {e}")
        return None

def fetch_evaluation(fen):
    evaluation_result = stockfish_api(fen, depth=18)  # Max depth 18

    if evaluation_result and "eval" in evaluation_result:
        return evaluation_result["eval"]

def fetch_best_move(fen):
    fetch_best_move = stockfish_api(fen, depth=18)  # Max depth 18

    if fetch_best_move and "move" in fetch_best_move:
        return fetch_best_move["move"]