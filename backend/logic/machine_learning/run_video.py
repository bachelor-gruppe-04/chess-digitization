import cv2
import onnxruntime as ort
import asyncio
from typing import Optional

from logic.machine_learning.game.game_store import GameStore
from logic.machine_learning.detection.run_detections import get_board_corners
from logic.machine_learning.board_state.map_pieces import get_payload
from logic.machine_learning.utilities.move import get_moves_pairs

async def process_video(
    video_path,
    piece_model_session: ort.InferenceSession,
    corner_ort_session: ort.InferenceSession,
    output_path: str,
    game_store: GameStore,
    game_id: str
) -> None:
    """
    Process a video file to detect a chessboard and map piece positions frame by frame.

    Args:
        video_path (str): Path to the input video.
        piece_model_session (ort.InferenceSession): ONNX runtime session for detecting chess pieces.
        corner_ort_session (ort.InferenceSession): ONNX runtime session for detecting board corners.
        output_path (str): Path to save the output video.
        game_store (GameStore): GameStore instance managing the state of ongoing games.
        game_id (str): Unique identifier for the game session.
    """
    cap: cv2.VideoCapture = cv2.VideoCapture(0)
    from logic.api.services.board_service import send_move
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return
    
    # # Show camera feed immediately (for debugging purposes)
    # while True:
    #     ret: bool
    #     video_frame: Optional[cv2.typing.MatLike]
    #     ret, video_frame = cap.read()
        
    #     if not ret or video_frame is None:
    #         print("Failed to capture frame.")
    #         break

    #     # Display the video frame as soon as it's captured
    #     cv2.imshow("Camera Feed", video_frame)

    #     # Wait for 1 ms before moving to the next frame
    #     key = cv2.waitKey(1) & 0xFF
    #     if key == ord('q'):  # Press 'q' to exit the camera feed
    #         break
    
    frame_counter: int = 0
    board_corners_ref: Optional[list] = None

    while cap.isOpened():
        ret: bool
        video_frame: Optional[cv2.typing.MatLike]
        ret, video_frame = cap.read()
        
        if not ret or video_frame is None:
            break

        if frame_counter % 5 == 0:
            if board_corners_ref is None:
                # Detect corners on the first frame
                board_corners_ref = await get_board_corners(video_frame, piece_model_session, corner_ort_session)
                if board_corners_ref is None:
                    print("Failed to detect corners.")
                    break

            game = game_store.get_game(game_id)
            if game:
                moves_pairs = get_moves_pairs(game.board)
                video_frame, payload = await get_payload(piece_model_session, video_frame, board_corners_ref, game, moves_pairs)
                if payload is not None:
                    print("Payload:", payload[1]["sans"])
                    
                    await send_move(1, payload[1]["sans"])
                    

            resized_frame = cv2.resize(video_frame, (1280, 720))
            cv2.imshow("Chess Board Detection", resized_frame)
            cv2.waitKey(1)

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()


async def prepare_to_run_video(video) -> None:
    """
    Main entry point for running the chessboard processing pipeline.
    Loads models, initializes game session, and starts video processing.
    """
    video_path = video
    output_path: str = 'resources/videoes/output_video_combined.avi'

    piece_model_path: str = "resources/models/480M_leyolo_pieces.onnx"
    corner_model_path: str = "resources/models/480L_leyolo_xcorners.onnx"

    piece_ort_session: ort.InferenceSession = ort.InferenceSession(piece_model_path)
    corner_ort_session: ort.InferenceSession = ort.InferenceSession(corner_model_path)

    game_store: GameStore = GameStore()
    game_id: str = "game_1"
    game_store.add_game(game_id)

    await process_video(video_path, piece_ort_session, corner_ort_session, output_path, game_store, game_id)


# if __name__ == "__main__":
#     asyncio.run(main())
