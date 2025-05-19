import asyncio
import threading
import logic.api.services.board_storage as storage

class BoardService:
  """ Service to manage chess boards and their operations. """
  
  def start_detectors(self) -> None:
    """ Start the chess detectors for all boards. """
    for board_id in storage.boards:
      thread = threading.Thread(
        target=self._run_detector_thread,
        args=(board_id,),
        daemon=True
      )
      thread.start()
      
  def _run_detector_thread(self, board_id: int) -> None:
    """ Run the detector in a separate thread. """
    asyncio.run(storage.boards[board_id].camera.detector.run())

  async def send_move(self, board_id: int, move: str) -> None:
    """ Send a chess move to all clients.

    Args:
      board_id (int): Board ID
      move (str): Chess move
    """
    board = storage.boards[board_id]

    checked_move, valid = board.validate_move(move)
    if valid:
      for client in board.clients:
        await client.send_text(checked_move)

  async def reset_game(self, board_id: int) -> None:
    """ Reset the chess game of a board. """
    board = storage.boards[board_id]
    for client in board.clients:
      await client.send_text(board.reset_board())

  async def reset_all_games(self) -> None:
    """ Reset the chess game to all boards. """
    for board_id in storage.boards:
      await self.reset_game(board_id)