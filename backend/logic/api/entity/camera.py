from typing import Generator
from .detector import Detector
import cv2

class Camera:
  """ Camera class to handle webcam. """
  
  def __init__(self, cam_id: int):
    """ Initialize the camera object.

    Args:
      cam_id (int): Camera ID
    """
    self.set_cam_id(cam_id)
    self.set_camera(cv2.VideoCapture(self.cam_id, cv2.CAP_DSHOW))
    self.detector = Detector(cam_id)
    
  def set_cam_id(self, cam_id: int) -> None:
    """ Set the camera ID.

    Args:
      cam_id (int): Camera ID
    Raises:
      TypeError: If the camera ID is not an integer.
      ValueError: If the camera ID is a negative integer.
    """
    if not isinstance(cam_id, int):
      raise TypeError("Camera ID must be an integer.")
    if cam_id < 1:
      raise ValueError("Camera ID must be a non-negative integer.")
    
    self.cam_id = cam_id
    
  def set_camera(self, camera: cv2.VideoCapture) -> None:
    """ Set the camera object.

    Args:
        camera (cv2.VideoCapture): Camera object
    """
    if not isinstance(camera, cv2.VideoCapture):
      raise TypeError("Camera must be a cv2.VideoCapture object.")
    if not camera.isOpened():
      raise CameraDoesNotExistError(f"Could not open Camera {self.cam_id}.")
    
    self.camera = camera
    
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
      
class CameraDoesNotExistError(Exception):
  """ Custom error class for camera errors. """
  def __init__(self, message: str):
    super().__init__(message)