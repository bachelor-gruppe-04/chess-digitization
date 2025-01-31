import requests

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

def analyze_position(fen):
    evaluation_result = stockfish_api(fen, depth=18)  # max depth 18

    if evaluation_result:
        if "eval" in evaluation_result:
            print(f"Stockfish evaluation: {evaluation_result['eval']}")

    return evaluation_result
