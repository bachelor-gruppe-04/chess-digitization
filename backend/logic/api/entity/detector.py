from logic.machine_learning.run_video import prepare_to_run_video
import cv2

class Detector:
  def __init__(self, id: int):#, video: cv2.VideoCapture):
    """Class to handle video processing for chessboard detection."""
    # self.video = video
    self.id = id
    
  async def run(self):
    detector_cap = cv2.VideoCapture(self.id, cv2.CAP_DSHOW)
    await prepare_to_run_video(self.id, detector_cap)
    