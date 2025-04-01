import cv2
import numpy as np
import onnxruntime as ort
import asyncio

from run_detections import find_scaled_labeled_board_corners, find_centers_of_squares
from find_FEN import find_fen
from render import render_centers
from typing import Optional, List, Tuple

#**IMPORTANT:** Break the loop by pressing 'Q'!!
async def process_video(
    video_path: str,
    piece_model_ref: ort.InferenceSession,
    corner_model_ref: ort.InferenceSession,
    output_path: str
) -> None:
    """
    Processes a video, detecting centers in the first frame, and saves the processed video.
    Then renders the centers in every frame.

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
    board_corners_initial = None  # Will hold the detected centers

    while cap.isOpened():
        ret: bool
        video_frame: Optional[np.ndarray]
        ret, video_frame = cap.read()

        if not ret or video_frame is None:
            break

        # Run find_centers only on the first frame
        if frame_counter == 0:
            board_corners = await find_scaled_labeled_board_corners(video_frame, piece_model_ref, corner_model_ref)
            if board_corners is not None:
                board_corners_initial = board_corners
            else:
                print("Failed to detect centers.")
                break

        if board_corners_initial is not None:
            
            fen = await find_fen(piece_model_ref,video_frame,board_corners)
            
            
            

            if frame_counter % 1 == 0:
                # resized_frame: np.ndarray = cv2.resize(centers2, (1280, 720))
                # cv2.imshow('Video', resized_frame)
                # out.write(centers2)
                print("ddd")

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()






async def main() -> None:
    video_path: str = 'resources/videoes/chessvideoKnights4.mp4'
    output_path: str = 'resources/videoes/output_video_combined.avi'

    piece_model_path: str = "resources/models/480M_leyolo_pieces.onnx"
    corner_model_path: str = "resources/models/480L_leyolo_xcorners.onnx"

    piece_ort_session: ort.InferenceSession = ort.InferenceSession(piece_model_path)
    corner_ort_session: ort.InferenceSession = ort.InferenceSession(corner_model_path)

    await process_video(video_path, piece_ort_session, corner_ort_session, output_path)

asyncio.run(main())
