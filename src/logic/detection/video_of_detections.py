import cv2
import numpy as np
import onnxruntime as ort
import asyncio
from run_detections import find_corners

#**IMPORTANT:** Break the loop by pressing 'Q'!!

async def process_video(video_path, piece_model_ref, corner_model_ref, output_path):
    """
    Processes a video, detecting corners in every 10th frame, and saves the processed video.

    Args:
        video_path (str): The path to the input video file.
        piece_model_ref (ort.InferenceSession): An ONNX Runtime InferenceSession for piece detection.
        corner_model_ref (ort.InferenceSession): An ONNX Runtime InferenceSession for corner detection.
        output_path (str): The path to save the processed video file.

    Returns:
        None.
    """
    # Starts VideoCapture object and prepares to read frames from the input video.
    cap = cv2.VideoCapture(video_path)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # FourCC compresses the video using the XVID codec.
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Initializes VideoWriter object to write processed frames to the output video.
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Initializes a frame counter to track the number of processed frames.
    frame_counter = 0

    # Main loop to process each frame of the video.
    while cap.isOpened():
        # Reads a frame from the input video.
        ret, video_frame = cap.read()

        if not ret:
            break

        # Processes every 10th frame to reduce processing load.
        if frame_counter % 10 == 0:

            # Finds corners in the current frame using the provided models.
            frame_with_centers = await find_corners(video_frame, piece_model_ref, corner_model_ref)

            if isinstance(frame_with_centers, np.ndarray):
                # Resizes the frame for display purposes.
                resized_frame = cv2.resize(frame_with_centers, (1280, 720))

                # Displays the resized frame in a window.
                cv2.imshow('Video', resized_frame)

                # Writes the processed frame to the output video file.
                out.write(frame_with_centers)
            else:
                print(f"Error: Expected a NumPy array, but got {type(frame_with_centers)}")

        frame_counter += 1

        # Exits the loop if the 'q' key is pressed.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


async def main():
    video_path = 'resources/videos/chessvideo.mp4'
    output_path = 'resources/videos/output_video_combined.avi'

    piece_model_path = "src/logic/models/480M_leyolo_pieces.onnx"
    corner_model_path = "src/logic/models/480L_leyolo_xcorners.onnx"

    # Loads models
    piece_ort_session = ort.InferenceSession(piece_model_path)
    corner_ort_session = ort.InferenceSession(corner_model_path)

    await process_video(video_path, piece_ort_session, corner_ort_session, output_path)

asyncio.run(main())