from logic.machine_learning.run_video import prepare_to_run_video
import cv2

class Detector:
  def __init__(self, video: cv2.VideoCapture):
    """Class to handle video processing for chessboard detection."""
    self.set_video(video)
    
  def set_video(self, video: cv2.VideoCapture) -> None:
    """Set the video capture object."""
    if not isinstance(video, cv2.VideoCapture):
      raise TypeError("Video must be a cv2.VideoCapture object.")
    self.video = video
    
  async def run(self):
    await prepare_to_run_video(self.video)
    