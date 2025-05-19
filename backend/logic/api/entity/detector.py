from logic.machine_learning.run_video import prepare_to_run_video
import cv2

class Detector:
  def __init__(self, id: int):
    """Class to handle video processing for chessboard detection."""
    self.set_id(id)
    
  def set_id(self, id: int) -> None:
    """ Set the ID of the detector. 
    
    Args:
      id (int): Detector ID
    Raises:
      TypeError: If the ID is not an integer.
      ValueError: If the ID is a negative integer.
    """
    if not isinstance(id, int):
      raise TypeError("ID must be an integer.")
    if id < 1:
      raise ValueError("ID must be a positive integer.")
    
    self.id = id
    
  async def run(self) -> None:
    detector_cap = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)
    await prepare_to_run_video(self.id, detector_cap)
    