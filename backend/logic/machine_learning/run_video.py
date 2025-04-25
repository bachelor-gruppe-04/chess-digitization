import time, cv2, onnxruntime as ort
from typing import Optional
from logic.machine_learning.game.game_store import GameStore
from logic.machine_learning.detection.run_detections import get_board_corners
from logic.machine_learning.board_state.map_pieces import get_payload
from logic.machine_learning.utilities.move import get_moves_pairs
import asyncio

# ──────────────────────────────────────────────────────────────
# tiny helper
def t(label: str, _last=[time.time()]):
    now = time.time()
    print(f"{label:<35} {(now - _last[0]) * 1000:7.1f} ms")
    _last[0] = now
# ──────────────────────────────────────────────────────────────


async def process_video(
    piece_model_session: ort.InferenceSession,
    corner_ort_session: ort.InferenceSession,
    game_store: GameStore,
    game_id: str,
    video: cv2.VideoCapture,
    board_id: int,
) -> None:

    t("Opening VideoCapture(0)")
    cap = video#cv2.VideoCapture(0, cv2.CAP_DSHOW)      # try CAP_DSHOW vs CAP_MSMF
    if not cap.isOpened():
        print("Error: Cannot open camera.")
        return
    t("cap.isOpened()")                           # << measure

    t("First cap.read()")
    ok, frame = cap.read()
    t("First frame grabbed")
    if not ok:
        print("Could not read first frame."); return

    frame_counter = 0
    board_corners_ref: Optional[list] = None
    from logic.api.services.board_service import send_move

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_counter % 5 == 0:
            if board_corners_ref is None:
                board_corners_ref = await get_board_corners(
                    frame, piece_model_session, corner_ort_session
                )
                t("get_board_corners")            # << measure
                if board_corners_ref is None:
                    print("Corners not found."); break

            game = game_store.get_game(game_id)
            if game:
                moves_pairs = get_moves_pairs(game.board)
                frame, payload = await get_payload(
                    piece_model_session, frame, board_corners_ref, game, moves_pairs
                )
                t("get_payload")                  # << measure
                if payload:
                    print("Payload:", payload)
                    await send_move(board_id, payload[1]["sans"])

            cv2.imshow("Chess Board Detection", cv2.resize(frame, (1280, 720)))
            cv2.waitKey(1)

        frame_counter += 1

    cap.release()
    cv2.destroyAllWindows()


async def prepare_to_run_video(board_id: int, video=None):
    t("Loading piece model")
    piece_session  = ort.InferenceSession("resources/models/480M_leyolo_pieces.onnx")
    t("Loading corner model")
    corner_session = ort.InferenceSession("resources/models/480L_leyolo_xcorners.onnx")

    game_store = GameStore(); game_id = "game_1"; game_store.add_game(game_id)
    await process_video(piece_session, corner_session, game_store, game_id, video, board_id)


# quick manual test
# if __name__ == "__main__":
#     asyncio.run(prepare_to_run_video())
