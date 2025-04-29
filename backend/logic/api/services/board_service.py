import asyncio
# from logic.api.entity.board import Board
# from logic.api.config import BOARD_COUNT
from logic.api.services.board_storage import boards

# boards = {i: Board(i) for i in range(1, (BOARD_COUNT + 1))}

class BoardService:

  async def start_detectors(self) -> None:
    for board_id in boards:
      asyncio.create_task(self.start_detector(board_id))
      
  async def start_detector(self, id:int) -> None:
    asyncio.create_task(boards[id].camera.detector.run())

  async def send_move(self, board_id: int, move: str):
    """ Send a chess move to all clients.

    Args:
      board_id (int): Board ID
      move (str): Chess move
    """
    board = boards[board_id]
    print(board.clients)
    checked_move, valid = board.validate_move(move)
    if valid:
      for client in board.clients:
        await client.send_text(checked_move)
        print(f"Move {move} sent to board {board_id}")

  async def reset_game(self, board_id: int):
    """ Reset the chess game of a board. """
    board = boards[board_id]
    for client in board.clients:
      await client.send_text(board.reset_board())
      print(f"reset_games() has been called")

  async def reset_all_games(self):
    """ Reset the chess game to all boards. """
    for board_id in boards:
      await self.reset_game(board_id)