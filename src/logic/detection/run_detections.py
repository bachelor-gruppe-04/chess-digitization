from constants import CORNER_KEYS, CORNER_MAPPING
# from render import render_corners
# from corner_slice import corners_set

from detection_methods import get_marker_xy
from corners_detection import run_xcorners_model, find_corners_from_xcorners, calculate_keypoints
from piece_detection import run_pieces_model

from detection_methods import get_keypoints
from warp import get_inv_transform, transform_centers, transform_boundary
from render import render_state



async def find3_corners(video_ref, dispatch, 
                         canvas_ref=None, set_text=None):
    

    video_height, video_width, _ = video_ref.shape


    pieces = await run_pieces_model(video_ref)
    black_pieces = [x for x in pieces if x[2] <= 5]
    white_pieces = [x for x in pieces if x[2] > 5]
    
    if len(black_pieces) == 0 or len(white_pieces) == 0:
        set_text(["No pieces to label corners"])
        return

    x_corners = await run_xcorners_model(video_ref)
    if len(x_corners) < 5:
        # With <= 5 xCorners, no quads are found
        set_text([f"Need ≥5 xCorners", f"Detected {len(x_corners)}"])
        return

    corners = find_corners_from_xcorners(x_corners)
    if corners is None:
        set_text(["Failed to find corners"])
        return


    keypoints: list[list[float]] = get_keypoints(corners, video_ref)


    centers = find_centers_of_squares(keypoints, video_ref)

    print("hdd")
    print(centers)


    # for key in CORNER_KEYS:
    #     xy = keypoints[CORNER_MAPPING[key]]
    #     payload = {
    #         "type": "SET_CORNER",
    #         "key": key,
    #         "xy": get_marker_xy(xy, video_height, video_width)
    #     }
    #     dispatch(payload)  # Use dispatch instead of direct function call
    #     print(payload)

    render_state2 = render_state(video_ref, centers)
    # set_text(["Found corners", "Ready to record"])

    return render_state2


def find_centers_of_squares(corners, canvas):

    
    keypoints = get_keypoints(corners, canvas)
    inv_transform = get_inv_transform(keypoints)
    centers, centers3D = transform_centers(inv_transform)
    boundary, boundary3D = transform_boundary(inv_transform)
    # boxes, scores = detect(pieces_model, video, keypoints)
    # squares = get_squares(boxes, centers3D, boundary3D)
    # state = get_update(scores, squares)
    # set_fen_from_state(state, color, dispatch, set_text)
    # render_state(centers, boundary)
    
    return centers
    # del boxes, scores, centers3D, boundary3D
