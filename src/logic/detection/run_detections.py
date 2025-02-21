from constants import CORNER_KEYS
from render import render_corners
# from corner_slice import corners_set

from detect import get_marker_xy
from corners_detection import run_xcorners_model, run_pieces_model, find_corners_from_xcorners, calculate_keypoints



async def find3_corners(video_ref, 
                         canvas_ref=None, dispatch=None, set_text=None):


    # pieces = await run_pieces_model(video_ref, pieces_model_ref)
    # black_pieces = [x for x in pieces if x[2] <= 5]
    # white_pieces = [x for x in pieces if x[2] > 5]
    
    # if len(black_pieces) == 0 or len(white_pieces) == 0:
    #     set_text(["No pieces to label corners"])
    #     return

    x_corners = await run_xcorners_model(video_ref)
    if len(x_corners) < 5:
        # With <= 5 xCorners, no quads are found
        set_text([f"Need ≥5 xCorners", f"Detected {len(x_corners)}"])
        return

    corners = find_corners_from_xcorners(x_corners)
    if corners is None:
        set_text(["Failed to find corners"])
        return
    

    print("XCORNERS")
    print(x_corners)
    print("CORNERS")
    print(corners)

    print("VIDEO REF")
    print(video_ref)

    # keypoints = calculate_keypoints(black_pieces, white_pieces, corners)

    # for key in CORNER_KEYS:
    #     xy = keypoints[key]
    #     payload = {
    #         "xy": get_marker_xy(xy, canvas_ref.height, canvas_ref.width),
    #         "key": key
    #     }
    #     dispatch(corners_set(payload))

    a = render_corners(video_ref, x_corners)
    # set_text(["Found corners", "Ready to record"])

    return a