import chess

from camera import Camera
from typing import List

class Board:
  
  def __init__(self, id: int):
    self.set_id(id)
    self.camera: Camera = Camera(id)
    self.move_history: List[str] = []
    self.chess_board: chess.Board = chess.Board()
  
  def set_id(self, id: int) -> TypeError | None:
    if type(id) != type(int):
      raise TypeError
      
    self.id = id
    
  def get_id(self) -> int:
    return self.id
    
  def get_camera(self) -> Camera:
    return self.camera
  
  def get_move_history(self) -> List[str]:
    return self.move_history
  
  def get_chess_board(self) -> chess.Board:
    return self.chess_board
  
  def move_piece(self, move: str) -> str:
    if type(move) != type(str):
      raise TypeError
    
    is_valid: bool = self.validate_move(move)
    command: str = ""
    
    if is_valid:
      self.move_history.append(move)
      command = move, is_valid
      
    else: 
      command = "INVALID", is_valid
      
    return command
      
  def validate_move(self, move: str) -> bool:
    if type(move) != type(str):
      raise TypeError
    
    is_valid: bool = None
    
    try:
      self.chess_board.push_san(move)
      is_valid = True
    except Exception: # chess.InvalidMoveError | chess.IllegalMoveError | chess.AmbiguousMoveError:
      is_valid = False
    
    return is_valid
  
  def reset_board(self) -> str:
    self.chess_board.reset()
    return "RESET"