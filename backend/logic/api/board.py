import chess
from fastapi import WebSocket
from camera import Camera
from typing import List, Literal

class Board:
  """ Chess board class to handle chess moves and history. """
  
  def __init__(self, id: int) -> None:
    """ Initialize the chess board object.
    
    Args:
      id (int): Board ID
    """
    self.id = id
    self.camera: Camera = Camera(id)
    self.move_history: List[str] = []
    self.clients: List[WebSocket] = []
    self.chess_board: chess.Board = chess.Board()
    self.invalid_latched: bool = False
  
  
  
  def process_move(self, move: str) -> (tuple[Literal['INVALID'], Literal[False]] | tuple[str, Literal[True]]):
    """ Process a chess move.
    
    Args:
      move (str): Chess move in SAN format.
    Returns:
      tuple[str, bool]: Tuple containing the move and a boolean indicating if the move was valid.
    """
    if not isinstance(move, str):
      raise TypeError(f"move must be a string, got {type(move).__name__}")
    
    move = move.strip()
    
    if self.invalid_latched:
      return "INVALID", False
  
    if self.validate_move(move):
      self.move_history.append(move)
      return move, True
    
    else: 
      self.invalid_latched = True
      return "INVALID", False
    
    
      
  def validate_move(self, move: str) -> bool:
    """ Validate a chess move.
    
    Args:
      move (str): Chess move in SAN format.
    Returns:
      bool: True if the move is valid, False otherwise.
    Raises:
      TypeError: If move is not a string.
    """
    if not isinstance(move, str):
      raise TypeError(f"move must be a string, got {type(move).__name__}")
    
    is_valid: bool = None
    
    try:
      self.chess_board.push_san(move)
      is_valid = True
    except Exception: # chess.InvalidMoveError | chess.IllegalMoveError | chess.AmbiguousMoveError:
      is_valid = False
    
    return is_valid
  
  
  
  def reset_board(self) -> str:
    """ Reset the chess board and move history.
    
    Returns:
      str: "RESET" if the board was reset successfully.
    """
    self.chess_board.reset()
    self.move_history = []
    self.invalid_latched = False
    return "RESET"