import chess
from fastapi import WebSocket
from camera import Camera
from typing import List

class Board:
  
  def __init__(self, id: int):
    self.set_id(id)
    self.camera: Camera = Camera(id)
    self.move_history: List[str] = []
    self.clients: List[WebSocket] = []
    self.chess_board: chess.Board = chess.Board()
    self.invalid_latched: bool = False
    
    
  
  def set_id(self, id: int) -> TypeError | None:
    if not isinstance(id, int):
      raise TypeError(f"id must be an integar, got {type(id).__name__}")
    if id < 0:
      raise ValueError(f"id must be a positive number, got {id}")
      
    self.id = id
    
    
    
  def get_id(self) -> int:
    return self.id
  
  
    
  def get_camera(self) -> Camera:
    return self.camera
  
  
  
  def get_move_history(self) -> List[str]:
    return self.move_history
  
  
  
  def get_chess_board(self) -> chess.Board:
    return self.chess_board
  
  
  
  def check_move(self, move: str):
    if not isinstance(move, str):
      raise TypeError(f"move must be a string, got {type(move).__name__}")
    
    if self.invalid_latched:
      return "INVALID", False
    
    is_valid: bool = self.validate_move(move)
  
    if is_valid:
      self.move_history.append(move)
      return move, True
    
    else: 
      self.invalid_latched = True
      return "INVALID", False
    
    
      
  def validate_move(self, move: str) -> bool:
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
    self.chess_board.reset()
    return "RESET"