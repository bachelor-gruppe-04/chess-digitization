from constants import CORNER_KEYS

from corners_detection import run_xcorners_model, find_corners_from_xcorners, calculate_keypoints
from piece_detection import run_pieces_model

from detection_methods import get_corners_of_chess_board, get_marker_xy
from warp import get_inv_transform, transform_centers, transform_boundary
from render import visualize_centers_opencv
from corner_slice import corners_set


async def find_corners(video_ref, pieces_model_ref, xcorners_model_ref):

    pieces = await run_pieces_model(video_ref, pieces_model_ref)
    black_pieces = [x for x in pieces if x[2] <= 5]
    white_pieces = [x for x in pieces if x[2] > 5]

    if len(black_pieces) == 0 or len(white_pieces) == 0:
        return

    x_corners = await run_xcorners_model(video_ref, xcorners_model_ref, pieces)

    # if len(x_corners) < 5:
    #     return

    corners = find_corners_from_xcorners(x_corners)


    keypoints = calculate_keypoints(black_pieces, white_pieces, corners)


    video_height, video_width, _ = video_ref.shape
    for key in CORNER_KEYS:
        xy = keypoints[key]
        payload = {
            "xy": get_marker_xy(xy, video_height, video_width),
            "key": key
        }
        print("Hhh")
        print(payload)
        # a = corners_set(payload)

    centers = find_centers_of_squares(corners, video_ref)
    
    a= visualize_centers_opencv(video_ref, corners)

    return a




def find_centers_of_squares(corners, video_ref):

    
    keypoints = get_corners_of_chess_board(corners, video_ref)
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
