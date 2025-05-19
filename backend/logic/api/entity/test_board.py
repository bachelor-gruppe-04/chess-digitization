import unittest
from unittest.mock import MagicMock
from logic.api.entity.board import Board

class TestBoard(unittest.TestCase):
  """ Unit tests for the Board class. """

  def test_initialization(self) -> None:
    """ Test the initialization of the Board class. """
    board = Board(1)
    
    self.assertEqual(board.id, 1)
    self.assertEqual(board.camera.cam_id, 1)
    self.assertEqual(board.camera.detector.id, 1)
    self.assertEqual(board.move_history, [])
    self.assertEqual(board.clients, [])
    self.assertFalse(board.invalid_latched)
    
  def test_invalid_value_initialization(self) -> None:
    """ Test the invalid value initialization of the Board class. """
    with self.assertRaises(ValueError):
      Board(0)
    
  def test_invalid_type_initialization(self) -> None:
    """ Test the invalid type initialization of the Board class. """
    with self.assertRaises(TypeError):
      Board("INVALID")
    
  def test_validate_move_one_valid(self) -> None:
    """ Test the validation of a single valid move. """
    board = Board(1)
    
    move, valid = board.validate_move("a4")
    self.assertEqual(move, "a4")
    self.assertTrue(valid)
    
  def test_validate_move_two_valid(self) -> None:
    """ Test the validation of two valid moves. """
    board = Board(1)
    
    move_1, valid_1 = board.validate_move("a4")
    self.assertEqual(move_1, "a4")
    self.assertTrue(valid_1)
    
    move_2, valid_2 = board.validate_move("b5")
    self.assertEqual(move_2, "b5")
    self.assertTrue(valid_2)
    
  def test_validate_move_one_invalid(self) -> None:
    """ Test the validation of a single invalid move. """
    board = Board(1)
    
    move, valid = board.validate_move("a5")
    self.assertEqual(move, "INVALID")
    self.assertFalse(valid)
    
  def test_validate_move_two_invalid(self) -> None:
    """ Test the validation of two invalid moves. """
    board = Board(1)
    
    move_1, valid_1 = board.validate_move("a5")
    self.assertEqual(move_1, "INVALID")
    self.assertFalse(valid_1)
    
    move_2, valid_2 = board.validate_move("b6")
    self.assertEqual(move_2, "INVALID")
    self.assertFalse(valid_2)
    
  def test_validate_move_first_valid_second_invalid(self) -> None:
    """ Test the validation of a valid move followed by an invalid move. """
    board = Board(1)
    
    move_1, valid_1 = board.validate_move("a4")
    self.assertEqual(move_1, "a4")
    self.assertTrue(valid_1)
    
    move_2, valid_2 = board.validate_move("Be4")
    self.assertEqual(move_2, "INVALID")
    self.assertFalse(valid_2)
    
  def test_validate_move_first_invalid_second_valid(self) -> None:
    """ Test the validation of an invalid move followed by a valid move. """
    board = Board(1)
    
    move_1, valid_1 = board.validate_move("a5")
    self.assertEqual(move_1, "INVALID")
    self.assertFalse(valid_1)
    
    move_2, valid_2 = board.validate_move("b5")
    self.assertEqual(move_2, "INVALID")
    self.assertFalse(valid_2)
    
  def test_reset_board(self) -> None:
    """ Test the reset_board method. """
    board = Board(1)
    
    self.assertEqual(board.reset_board(), "RESET")
    
  def test_reset_board_valid_game(self) -> None:
    """ Test the reset_board method with a valid game. """
    board = Board(1)
    
    move_11, valid_11 = board.validate_move("a4")
    self.assertEqual(move_11, "a4")
    self.assertTrue(valid_11)
    
    move_12, valid_12 = board.validate_move("b5")
    self.assertEqual(move_12, "b5")
    self.assertTrue(valid_12)
    
    board.reset_board()
    
    move_21, valid_21 = board.validate_move("a4")
    self.assertEqual(move_21, "a4")
    self.assertTrue(valid_21)
    
    move_22, valid_22 = board.validate_move("b5")
    self.assertEqual(move_22, "b5")
    self.assertTrue(valid_22)
    
  def test_reset_board_invalid_game_once(self) -> None:
    """ Test the reset_board method with an invalid game. """
    board = Board(1)
    
    move_11, valid_11 = board.validate_move("a4")
    self.assertEqual(move_11, "a4")
    self.assertTrue(valid_11)
    
    move_12, valid_12 = board.validate_move("Be4")
    self.assertEqual(move_12, "INVALID")
    self.assertFalse(valid_12)
    
    board.reset_board()
    
    move_21, valid_21 = board.validate_move("a4")
    self.assertEqual(move_21, "a4")
    self.assertTrue(valid_21)
    
    move_22, valid_22 = board.validate_move("b5")
    self.assertEqual(move_22, "b5")
    self.assertTrue(valid_22)
    
  def test_reset_board_invalid_game_twice(self) -> None:
    """ Test the reset_board method with an invalid game twice. """
    board = Board(1)
    
    move_11, valid_11 = board.validate_move("a4")
    self.assertEqual(move_11, "a4")
    self.assertTrue(valid_11)
    
    move_12, valid_12 = board.validate_move("Be4")
    self.assertEqual(move_12, "INVALID")
    self.assertFalse(valid_12)
    
    board.reset_board()
    
    move_21, valid_21 = board.validate_move("a5")
    self.assertEqual(move_21, "INVALID")
    self.assertFalse(valid_21)
    
    move_22, valid_22 = board.validate_move("b5")
    self.assertEqual(move_22, "INVALID")
    self.assertFalse(valid_22)
    
    board.reset_board()
    
    move_31, valid_31 = board.validate_move("a4")
    self.assertEqual(move_31, "a4")
    self.assertTrue(valid_31)
    
    move_32, valid_32 = board.validate_move("b5")
    self.assertEqual(move_32, "b5")
    self.assertTrue(valid_32)
    
  def test_reset_board_forced_failure(self) -> None:
    """ Test the reset_board method with a forced failure. """
    board = Board(1)
      
    board.chess_board.reset = MagicMock(side_effect=Exception("forced failure"))
      
    self.assertEqual(board.reset_board(), "RESET_FAILED")