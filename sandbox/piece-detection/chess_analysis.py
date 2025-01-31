import requests

def stockfish_api(fen, depth):
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
    
def analyze_position(fen):
    evaluation_result = stockfish_api(fen, depth=15)  # Adjust depth as needed

    if evaluation_result:
        print(evaluation_result)
        if "evaluation" in evaluation_result:
            print(f"Stockfish evaluation: {evaluation_result['evaluation']}")

    return evaluation_result