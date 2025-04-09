# from constants import START_FEN

# class Game:
#     def __init__(self):
#         self.moves = ""
#         self.fen = START_FEN
#         self.start = START_FEN
#         self.last_move = ""
#         self.greedy = False



# class GameSlice:
#     def __init__(self):
#         self.state = Game()

#     def game_set_moves(self, moves: str):
#         self.state.moves = moves

#     def game_set_fen(self, fen: str):
#         self.state.fen = fen

#     def game_set_start(self, start: str):
#         self.state.start = start

#     def game_set_last_move(self, last_move: str):
#         self.state.last_move = last_move

#     def game_reset_moves(self):
#         self.state.moves = ""

#     def game_reset_fen(self):
#         self.state.fen = START_FEN

#     def game_reset_start(self):
#         self.state.start = START_FEN

#     def game_reset_last_move(self):
#         self.state.last_move = ""

#     def game_update(self, payload: Dict[str, Optional[str]]):
#         self.state.moves = payload.get("moves", self.state.moves)
#         self.state.fen = payload.get("fen", self.state.fen)
#         self.state.last_move = payload.get("lastMove", self.state.last_move)
#         self.state.greedy = payload.get("greedy", self.state.greedy)