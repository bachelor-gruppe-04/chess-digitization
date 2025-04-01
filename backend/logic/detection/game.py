import chess

class GameStore:
    def __init__(self):
        self.games = {}  # Dictionary storing multiple games centrally

    def add_game(self, game_id):
        if game_id not in self.games:
            self.games[game_id] = {
                "board": chess.Board(),
                "moves_pairs": [],
                "last_move": None
            }

    def update_game(self, game_id, move):
        if game_id in self.games:
            self.games[game_id]["last_move"] = move
            self.games[game_id]["moves_pairs"].append(move)

    def get_game(self, game_id):
        return self.games.get(game_id, None)

# Usage:
game_store = GameStore()
game_store.add_game("game_1")
game_store.update_game("game_1", "e2e4")

print(game_store.get_game("game_1")["last_move"])  # Output: e2e4
