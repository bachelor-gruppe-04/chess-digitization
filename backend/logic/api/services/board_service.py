import asyncio
from logic.api.entity.board import Board
from logic.api.config import BOARD_COUNT

boards = {i: Board(i) for i in range(1, (BOARD_COUNT + 1))}

async def start_detector() -> None:
  for board_id in boards:
    print(f"Starting detector for board {board_id}")
    print(boards[board_id])
    print(boards[board_id].camera.detector)
    asyncio.create_task(boards[board_id].camera.detector.run())
    

async def send_move(board_id: int, move: str):
  """ Send a chess move to all clients.

  Args:
    board_id (int): Board ID
    move (str): Chess move
  """
  board = boards[board_id]
  checked_move, valid = board.validate_move(move)
  if valid:
    for client in board.clients:
      await client.send_text(checked_move)

async def reset_game(board_id: int):
  """ Reset the chess game of a board. """
  board = boards[board_id]
  for client in board.clients:
    await client.send_text(board.reset_board())

async def reset_all_games():
  """ Reset the chess game to all boards. """
  for board_id in boards:
    await reset_game(board_id)