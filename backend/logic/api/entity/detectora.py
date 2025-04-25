from logic.machine_learning.run_video import prepare_to_run_video
import cv2

class Detector:
  def __init__(self, id: int, video: cv2.VideoCapture):
    """Class to handle video processing for chessboard detection."""
    self.video = video
    self.id = id
    
  async def run(self):
    await prepare_to_run_video(self.id, self.video)
    