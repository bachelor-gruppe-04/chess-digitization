import cv2
import numpy as np
import onnxruntime as ort
import asyncio
from game_store import GameStore
from run_detections import find_scaled_labeled_board_corners
from map_pieces import find_pieces
from moves import get_moves_pairs

async def process_video(video_path, piece_model_session, corner_ort_session, output_path, game_store, game_id):
    """Main processing loop for the video (equivalent to React's useEffect on load)."""
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video source.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_counter = 0
    board_corners_ref = None

    while cap.isOpened():
        ret, video_frame = cap.read()
        if not ret or video_frame is None:
            break

        # Only process every 40th frame
        if frame_counter % 5 == 0:
            if board_corners_ref is None:
                # Detect corners and set up the game board
                board_corners_ref = await find_scaled_labeled_board_corners(video_frame, piece_model_session, corner_ort_session)
                if board_corners_ref is None:
                    print("Failed to detect corners.")
                    break

            # Get the game object and process the moves
            game = game_store.get_game(game_id)
            if game:
                moves_pairs = get_moves_pairs(game.board)
                # Now we send both game and moves_pairs to find_pieces and get the updated frame
                video_frame = await find_pieces(piece_model_session, video_frame, board_corners_ref, game, moves_pairs)
                
            resized_frame = cv2.resize(video_frame, (1280, 720))
            cv2.imshow("Chess Board Detection", resized_frame)
            cv2.waitKey(1)

        # Increment the frame counter
        frame_counter += 1

    cap.release()
    out.release()
    cv2.destroyAllWindows()



async def main() -> None:
        
    video_path: str = 'resources/videoes/new/TopViewWhite.mp4'  # Path to your prerecorded video
    output_path: str = 'resources/videoes/output_video_combined.avi'

    piece_model_path: str = "resources/models/480M_leyolo_pieces.onnx"
    corner_model_path: str = "resources/models/480L_leyolo_xcorners.onnx"

    piece_ort_session: ort.InferenceSession = ort.InferenceSession(piece_model_path)
    corner_ort_session: ort.InferenceSession = ort.InferenceSession(corner_model_path)

    # Assuming game_store and game_id are defined somewhere
    # For example:
    game_store = GameStore()
    game_id = "game_1"  # Each video gets a unique game ID
    game_store.add_game(game_id)

    # Specify whether you want to use prerecorded video or live video

    await process_video(video_path, piece_ort_session, corner_ort_session, output_path, game_store, game_id)

asyncio.run(main())
