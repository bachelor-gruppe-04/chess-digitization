import chess

class GameStore:
    def __init__(self):
        # Dictionary storing multiple games centrally
        self.games = {}  

    def add_game(self, game_id):
        if game_id not in self.games:
            self.games[game_id] = {
                "board": chess.Board(),  # Initializes the chess board
                "moves_pairs": [],       # Stores the move pairs
                "last_move": None         # Tracks the last move
            }

    def update_game(self, game_id, move):
        if game_id in self.games:
            # Update the last move in the game
            self.games[game_id]["last_move"] = move
            # Append the move to the move pairs list
            self.games[game_id]["moves_pairs"].append(move)

    def get_game(self, game_id):
        return self.games.get(game_id, None)

    def get_move_history(self, game_id):
        game = self.get_game(game_id)
        if game:
            history = game["board"].history()
            if len(history) == 0:
                return ""
            elif len(history) == 1:
                return f"1. {history[-1]}"
            else:
                first_move = history[-2]
                second_move = history[-1]
                n_half_moves = len(history) // 2
                if len(history) % 2 == 0:
                    return f"{n_half_moves}.{first_move} {second_move}"
                return f"{n_half_moves}...{first_move} {n_half_moves + 1}.{second_move}"
        return ""

    def get_moves_pairs(self, game_id):
        game = self.get_game(game_id)
        if game:
            return game["moves_pairs"]
        return []

    def get_last_move(self, game_id):
        game = self.get_game(game_id)
        if game:
            return game["last_move"]
        return None