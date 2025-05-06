import asyncio
import threading
import chess
from logic.api.services.board_storage import boards

class BoardService:
  
  def start_detectors(self) -> None:
    """ Start the chess detectors for all boards. """
    for board_id in boards:
      thread = threading.Thread(
        target=self._run_detector_thread,
        args=(board_id,),
        daemon=True
      )
      thread.start()
      
  def _run_detector_thread(self, board_id: int):
    """ Run the detector in a separate thread. """
    asyncio.run(boards[board_id].camera.detector.run())

  async def send_move(self, board_id: int, move: str):
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

  async def reset_game(self, board_id: int):
        """ Reset the chess game of a board. """
        board = boards[board_id]

        print("THIS IS INSIDE BOARD SERVICE")
        print(board.chess_board)
        board.chess_board = chess.Board()
        print(board.chess_board)
        reset_message = board.reset_board() 
        
        for client in board.clients:
            await client.send_text(reset_message)

  async def reset_all_games(self):
    """ Reset the chess game to all boards. """
    for board_id in boards:
      await self.reset_game(board_id)