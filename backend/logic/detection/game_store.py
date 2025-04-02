from game import Game

class GameStore:
    def __init__(self):
        self.games = {}  # Dictionary to store multiple Game instances

    def add_game(self, game_id: str):
        """Adds a new game if it does not already exist."""
        if game_id not in self.games:
            self.games[game_id] = Game(game_id)

    def update_game(self, game_id: str, move: str):
        """Updates an existing game by making a move."""
        if game_id in self.games:
            game = self.games[game_id]
            game.update_last_move(move)

    def get_game(self, game_id: str):
        """Retrieves a game instance by ID."""
        return self.games.get(game_id, None)

    def get_move_history(self, game_id: str) -> str:
        """Formats the move history into a readable string."""
        game = self.get_game(game_id)
        if game:
            moves = game.get_moves_pairs()
            result = []
            for i in range(0, len(moves), 2):
                if i + 1 < len(moves):
                    result.append(f"{i // 2 + 1}. {moves[i]} {moves[i + 1]}")
                else:
                    result.append(f"{i // 2 + 1}. {moves[i]}")
            return " ".join(result)
        return ""

    def get_last_move(self, game_id: str):
        """Gets the last move of a specific game."""
        game = self.get_game(game_id)
        return game.last_move if game else None
