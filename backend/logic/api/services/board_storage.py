from logic.api.entity.board import Board

# boards = {}
BOARD_COUNT = 1
boards = {i: Board(i) for i in range(1, (BOARD_COUNT + 1))}