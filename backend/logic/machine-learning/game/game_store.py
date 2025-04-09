from game import Game

class GameStore:
    def __init__(self):
        self.games = {}  # Dictionary to store multiple Game instances

    def add_game(self, game_id: str):
        """Adds a new game if it does not already exist."""
        if game_id not in self.games:
            self.games[game_id] = Game(game_id)

    def get_game(self, game_id: str):
        """Retrieves a game instance by ID."""
        return self.games.get(game_id, None)