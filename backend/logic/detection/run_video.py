import cv2
import numpy as np
import onnxruntime as ort
import asyncio

from run_detections import find_corners
from typing import Optional

#**IMPORTANT:** Break the loop by pressing 'Q'!!

async def process_video(
    video_path: str,
    piece_model_ref: ort.InferenceSession,
    corner_model_ref: ort.InferenceSession,
    output_path: str
) -> None:
    """
    Processes a video, detecting corners in every 10th frame, and saves the processed video.

    Args:
        video_path (str): The path to the input video file.
        piece_model_ref (ort.InferenceSession): An ONNX Runtime InferenceSession for piece detection.
        corner_model_ref (ort.InferenceSession): An ONNX Runtime InferenceSession for corner detection.
        output_path (str): The path to save the processed video file.

    Returns:
        None
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)

    frame_width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps: float = cap.get(cv2.CAP_PROP_FPS)

    fourcc: int = cv2.VideoWriter_fourcc(*'XVID')
    out: cv2.VideoWriter = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_counter: int = 0

    while cap.isOpened():
        ret: bool
        video_frame: Optional[np.ndarray]
        ret, video_frame = cap.read()

        if not ret or video_frame is None:
            break

        if frame_counter % 10 == 0:
            frame_with_centers: Optional[np.ndarray] = await find_corners(video_frame, piece_model_ref, corner_model_ref)

            if isinstance(frame_with_centers, np.ndarray):
                resized_frame: np.ndarray = cv2.resize(frame_with_centers, (1280, 720))
                cv2.imshow('Video', resized_frame)
                out.write(frame_with_centers)
            else:
                print(f"Error: Expected a NumPy array, but got {type(frame_with_centers)}")

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()



async def main() -> None:
    video_path: str = 'backend/resources/videoes/chessvideo.mp4'
    output_path: str = 'backend/resources/videoes/output_video_combined.avi'

    piece_model_path: str = "backend/resources/models/480M_leyolo_pieces.onnx"
    corner_model_path: str = "backend/resources/models/480L_leyolo_xcorners.onnx"

    piece_ort_session: ort.InferenceSession = ort.InferenceSession(piece_model_path)
    corner_ort_session: ort.InferenceSession = ort.InferenceSession(corner_model_path)

    await process_video(video_path, piece_ort_session, corner_ort_session, output_path)

asyncio.run(main())
