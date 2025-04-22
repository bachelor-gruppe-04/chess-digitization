from typing import Generator
from .detectora import Detector
import cv2
import asyncio

class Camera:
  """ Camera class to handle webcam. """
  
  def __init__(self, cam_id: int):
    """ Initialize the camera object.

    Args:
      cam_id (int): Camera ID
    """
    self.cam_id = cam_id
    self.camera = cv2.VideoCapture(cam_id)
    self.detector = Detector(self.camera)
    
      
  def generate_frames(self) -> Generator[bytes, None, None]:
    """ Generate frames from the laptop webcam.
  
    Yields:
      Generator[bytes, None, None]: Image frames
    """
    while True:
      success, frame = self.camera.read()
      if not success:
        break
      _, buffer = cv2.imencode(".jpg", frame)
      yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")