import cv2
import numpy as np
import onnxruntime as ort
import asyncio
from run_detections import find_corners
from corner_slice import corners_set

def dispatch(action):
    """Handles dispatched actions and updates the state."""
    if action["type"] == "SET_CORNER":
        corners_set(action["key"], action["xy"])
    else:
        raise ValueError(f"Unknown action type: {action['type']}")
    


async def process_video(video_path, piece_model_ref, corner_model_ref, output_path):
    cap = cv2.VideoCapture(video_path)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_counter = 0

    while cap.isOpened():
        ret, video_frame = cap.read()
        if not ret:
            break

        if frame_counter % 1 == 0:
            # Call the async function to process the frame
            result_frame = await find_corners(video_frame, piece_model_ref, corner_model_ref)
            print(result_frame)



            if isinstance(result_frame, np.ndarray):  # Ensure the frame is valid
                resized_frame = cv2.resize(result_frame, (1280, 720))
                cv2.imshow('Video', resized_frame)
                out.write(result_frame)
            else:
                print(f"Error: Expected a NumPy array, but got {type(result_frame)}")

        frame_counter += 1

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

    piece_ort_session = ort.InferenceSession(piece_model_path)
    corner_ort_session = ort.InferenceSession(corner_model_path)

    await process_video(video_path, piece_ort_session, corner_ort_session, output_path)

# Run the async main function
asyncio.run(main())