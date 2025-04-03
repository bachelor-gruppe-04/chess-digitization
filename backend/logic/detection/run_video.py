import cv2
import numpy as np
import onnxruntime as ort
import asyncio
import chess
from game import Game
from game_store import GameStore
from run_detections import find_scaled_labeled_board_corners, find_centers_and_boundary
from map_pieces import find_pieces
from moves import get_moves_pairs
from typing import Optional, List, Tuple
from render import draw_points, draw_polygon

async def process_video(video_path, piece_model_session, corner_ort_session, output_path, game_store, game_id, live_video=False):
    """Main processing loop for the video (equivalent to React's useEffect on load)."""
    
    cap = cv2.VideoCapture(0) if live_video else cv2.VideoCapture(video_path)
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

        if frame_counter == 0:
            board_corners_ref = await find_scaled_labeled_board_corners(video_frame, piece_model_session, corner_ort_session)
            print(board_corners_ref)
            if board_corners_ref is None:
                print("Failed to detect centers.")
                break
            
            game = game_store.get_game(game_id)
            if game:
                # Update game state before processing
                moves_pairs = get_moves_pairs(game.board)  # Update possible moves
                last_move = game.last_move  # Get the last move
                
                # Store the updated game state
                game_store.update_game(last_move, game_id)
                        
            
            centers, boundary = find_centers_and_boundary(board_corners_ref, video_frame)  # Find centers of squares
                        
            frame = draw_points(video_frame, centers)  # Draw centers on the frame
            frame2 = draw_polygon(frame, boundary)  # Draw boundary on the frame
                        
            resized_frame = cv2.resize(frame2, (1280, 720))

            
            # Show the frame with detected centers
            cv2.imshow("Chess Board Detection", resized_frame)
            cv2.waitKey(1)  # Refresh the display
            
            await find_pieces(piece_model_session, video_frame, board_corners_ref, game_store.get_game(game_id), moves_pairs)

            
        frame_counter += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
                
                
        


async def main() -> None:
    video_path: str = 'resources/videoes/chessvideoKnights4.mp4'  # Path to your prerecorded video
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
    live_video = False  # Set to True for live video processing

    await process_video(video_path, piece_ort_session, corner_ort_session, output_path, game_store, game_id, live_video)

asyncio.run(main())


# def update_game_state(game_store, game_id):
#     """Updates the game state whenever the game changes."""
#     game = game_store.get_game(game_id)
    
#     if not game:
#         return
    
#     # Recreate the board
#     board = make_board(game)  
#     move_text = get_move_text(board)  # Equivalent to `moveTextRef.current = getMoveText(board)`

#     # Handle greedy logic
#     if game.get("greedy", False):
#         board.undo()
#     else:
#         moves_pairs = get_moves_pairs(board)  # Equivalent to `movesPairsRef.current = getMovesPairs(board)`

#     # Update the game store with new values
#     game["board"] = board
#     game["last_move"] = game.get("last_move", None)
#     game["moves_pairs"] = moves_pairs

#     # Save the updated game
#     game_store.update_game(game_id, game["last_move"])

#     print(f"Updated game state: {game}")
#     return game

