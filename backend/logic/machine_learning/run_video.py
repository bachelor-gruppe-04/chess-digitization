import cv2
import onnxruntime as ort
import asyncio
from typing import Optional

from game.game_store import GameStore
from detection.run_detections import get_board_corners
from board_state.map_pieces import get_payload
from utilities.move import get_moves_pairs


async def process_video(
    video_path: str,
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
    cap: cv2.VideoCapture = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    frame_width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps: float = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out: cv2.VideoWriter = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
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

            resized_frame = cv2.resize(video_frame, (1280, 720))
            cv2.imshow("Chess Board Detection", resized_frame)
            cv2.waitKey(1)

        frame_counter += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()


async def main() -> None:
    """
    Main entry point for running the chessboard processing pipeline.
    Loads models, initializes game session, and starts video processing.
    """
    video_path: str = 'resources/videoes/new/TopViewWhite.mp4'
    output_path: str = 'resources/videoes/output_video_combined.avi'

    piece_model_path: str = "resources/models/480M_leyolo_pieces_simplified.onnx"
    corner_model_path: str = "resources/models/480L_leyolo_xcorners.onnx"

    piece_ort_session: ort.InferenceSession = ort.InferenceSession(piece_model_path)
    corner_ort_session: ort.InferenceSession = ort.InferenceSession(corner_model_path)

    game_store: GameStore = GameStore()
    game_id: str = "game_1"
    game_store.add_game(game_id)

    await process_video(video_path, piece_ort_session, corner_ort_session, output_path, game_store, game_id)


if __name__ == "__main__":
    asyncio.run(main())
