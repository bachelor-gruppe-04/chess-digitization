from game import GameStore
import cv2
import numpy as np
import onnxruntime as ort
import asyncio
import chess

from run_detections import find_scaled_labeled_board_corners, find_centers_of_squares
from map_pieces import find_pieces
from moves import get_moves_pairs
from find_FEN import find_fen
from render import render_centers
from typing import Optional, List, Tuple


async def process_video(video_path, piece_ort_session, corner_ort_session, output_path, game_store, game_id):
    """Main processing loop for the video (equivalent to React's useEffect on load)."""
    
    cap = cv2.VideoCapture(video_path)
    
    # Get video frame details
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    frame_counter = 0
    board_corners_initial = None
    last_move = None
    
    while cap.isOpened():
        ret, video_frame = cap.read()
        if not ret or video_frame is None:
            break

        # First frame: Detect board corners (like React's setup logic)
        if frame_counter == 0:
            board_corners_initial = await find_scaled_labeled_board_corners(video_frame, piece_ort_session, corner_ort_session)
            if board_corners_initial is None:
                print("Failed to detect centers.")
                break
            print(board_corners_initial)

        # Process each frame (similar to React's findPieces)
        if board_corners_initial is not None:
            # Retrieve last move from GameStore
            game = game_store.get_game(game_id)
            last_move = game["last_move"] if game else None
            print(f"Last Move: {last_move}")

            # Find pieces on the board (and simulate new move detection)
            playing_ref = True
            moves_pairs = get_moves_pairs(game["board"])
            find_pieces(piece_ort_session, video_frame, playing_ref, board_corners_initial, game["board"], moves_pairs)

            # Optionally render or save frames (like React's rendering)
            if frame_counter % 1 == 0:
                print(f"Processing frame {frame_counter}")
                out.write(video_frame)

            # After processing, simulate a move update
            new_move = "e2e4"  # This would come from actual move detection (e.g., from the video)
            game_store.update_game(game_id, new_move)

        frame_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Key press 'Q' to quit
            break

    cleanup(cap, out)


def cleanup(cap, out):
    """Cleanup method to close streams (similar to React cleanup in useEffect)."""
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

    # Assuming game_store and game_id are defined somewhere
    # For example:
    game_store = GameStore()
    game_id = "game_1"
    game_store.add_game(game_id)

    await process_video(video_path, piece_ort_session, corner_ort_session, output_path, game_store, game_id)

asyncio.run(main())
