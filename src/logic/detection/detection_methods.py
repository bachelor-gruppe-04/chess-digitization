from constants import MODEL_WIDTH, MODEL_HEIGHT, MARKER_DIAMETER, CORNER_KEYS

import numpy as np
import tensorflow as tf
from preprocess import preprocess_image


async def process_boxes_and_scores(boxes, scores):
    max_scores = tf.reduce_max(scores, axis=1)
    argmax_scores = tf.argmax(scores, axis=1)
    nms = tf.image.non_max_suppression(boxes, max_scores, max_output_size=100, iou_threshold=0.3, score_threshold=0.1)
    
    # Use get_centers function to get the centers from the selected boxes
    centers = get_centers_of_bbox(tf.gather(boxes, nms, axis=0))

    # Gather the class indices of the selected boxes and expand dimensions
    cls = tf.expand_dims(tf.gather(argmax_scores, nms, axis=0), axis=1)

    # Cast cls to float16 (ensure it's compatible with centers)
    cls = tf.cast(cls, dtype=tf.float16)

    # Concatenate the centers with the class indices
    res = tf.concat([centers, cls], axis=1)

    res_array = res.numpy()
    
    return res_array

def get_boxes_and_scores(preds, width, height, video_width, video_height, padding, roi):
    # preds is assumed to be a NumPy array with shape (batch_size, num_boxes, num_predictions)
    preds_t = np.transpose(preds, (0, 2, 1))  # Transpose preds to match the desired shape

    # Extract width (w) and height (h)
    w = preds_t[:, :, 2:3]  # Shape: (batch_size, num_boxes, 1)
    h = preds_t[:, :, 3:4]  # Shape: (batch_size, num_boxes, 1)
    
    # xc, yc, w, h -> l, t, r, b (left, top, right, bottom)
    l = preds_t[:, :, 0:1] - (w / 2)  # Left
    t = preds_t[:, :, 1:2] - (h / 2)  # Top
    r = l + w  # Right
    b = t + h  # Bottom

    l = l - padding[0]
    r = r - padding[0]
    t = t - padding[2]
    b = b - padding[2]

    # Scale the bounding box coordinates
    l = l * (width / (MODEL_WIDTH - padding[0] - padding[1]))
    r = r * (width / (MODEL_WIDTH - padding[0] - padding[1]))
    t = t * (height / (MODEL_HEIGHT - padding[2] - padding[3]))
    b = b * (height / (MODEL_HEIGHT - padding[2] - padding[3]))

    # Add ROI
    l = l + roi[0]
    r = r + roi[0]
    t = t + roi[1]
    b = b + roi[1]

    # Scale based on video size
    l = l * (MODEL_WIDTH / video_width)
    r = r * (MODEL_WIDTH / video_width)
    t = t * (MODEL_HEIGHT / video_height)
    b = b * (MODEL_HEIGHT / video_height)

    # Concatenate the left, top, right, and bottom coordinates to form the bounding boxes
    boxes = np.concatenate([l, t, r, b], axis=2)  # Shape: (batch_size, num_boxes, 4)

    # Extract the scores (assuming score is in the 5th element onward)
    scores = preds_t[:, :, 4:]  # Shape: (batch_size, num_boxes, num_classes)

    # Squeeze to remove unnecessary dimensions (if any)
    boxes = np.squeeze(boxes, axis=0)
    scores = np.squeeze(scores, axis=0)
    return boxes, scores


def get_center(points):
    center = [sum(x[0] for x in points), sum(x[1] for x in points)]
    center = [x / len(points) for x in center]
    return center

def euclidean(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    return dist

def get_input(video_ref, keypoints=None, padding_ratio=12):
    video_height, video_width, _ = video_ref.shape
    roi = None

    if keypoints is not None:
        bbox = get_bbox(keypoints)
        padding_left = int(bbox['width'] / padding_ratio)
        padding_right = int(bbox['width'] / padding_ratio)
        padding_top = int(bbox['height'] / padding_ratio)
        padding_bottom = int(bbox['height'] / padding_ratio)

        padded_roi_width = bbox['width'] + padding_left + padding_right
        padded_roi_height = bbox['height'] + padding_top + padding_bottom
        ratio = padded_roi_height / padded_roi_width
        desired_ratio = MODEL_HEIGHT / MODEL_WIDTH

        if ratio > desired_ratio:
            target_width = padded_roi_height / desired_ratio
            dx = target_width - padded_roi_width
            padding_left += int(dx / 2)
            padding_right += dx - int(dx / 2)
        else:
            target_height = padded_roi_width * desired_ratio
            padding_top += target_height - padded_roi_height

        roi = [
            max(int(video_width * (bbox['xmin'] - padding_left) / MODEL_WIDTH), 0),
            max(int(video_height * (bbox['ymin'] - padding_top) / MODEL_HEIGHT), 0),
            min(int(video_width * (bbox['xmax'] + padding_right) / MODEL_WIDTH), video_width),
            min(int(video_height * (bbox['ymax'] + padding_bottom) / MODEL_HEIGHT), video_height)
        ]
    else:
        roi = [0, 0, video_width, video_height]

    # Cropping
    video_ref = video_ref[roi[1]:roi[3], roi[0]:roi[2], :]
    
    # Resizing
    height, width, _ = video_ref.shape
    ratio = height / width
    desired_ratio = MODEL_HEIGHT / MODEL_WIDTH
    resize_height = MODEL_HEIGHT
    resize_width = MODEL_WIDTH
    if ratio > desired_ratio:
        resize_width = int(MODEL_HEIGHT / ratio)
    else:
        resize_height = int(MODEL_WIDTH * ratio)
    
    video_ref = tf.image.resize(video_ref, [resize_height, resize_width])
    
    # Padding
    dx = MODEL_WIDTH - video_ref.shape[1]
    dy = MODEL_HEIGHT - video_ref.shape[0]
    pad_right = dx // 2
    pad_left = dx - pad_right
    pad_bottom = dy // 2
    pad_top = dy - pad_bottom
    padding = [pad_left, pad_right, pad_top, pad_bottom]

    image4d = preprocess_image(video_ref.numpy())


    
    return image4d, width, height, padding, roi

def extract_xy_from_corners_mapping(corners_mapping, canvas_ref):
    canvas_height, canvas_width, _ = canvas_ref.shape
    return [get_xy(corners_mapping[x]['xy'], canvas_height, canvas_width) for x in CORNER_KEYS]


def get_xy(marker_xy, height, width):
    sx = MODEL_WIDTH / width
    sy = MODEL_HEIGHT / height
    xy = [sx * marker_xy[0], sy * (marker_xy[1] + height + MARKER_DIAMETER)]
    return xy



def get_bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    
    xmin = min(xs)
    xmax = max(xs)
    ymin = min(ys)
    ymax = max(ys)

    width = xmax - xmin
    height = ymax - ymin

    bbox = {
        "xmin": xmin,
        "xmax": xmax,
        "ymin": ymin,
        "ymax": ymax,
        "width": width,
        "height": height
    }

    return bbox


def scale_labeled_board_corners(xy, height, width):
    sx = width / MODEL_WIDTH
    sy = height / MODEL_HEIGHT
    marker_xy = [sx * xy[0], sy * xy[1] - height - MARKER_DIAMETER]
    return marker_xy


def get_centers_of_bbox(boxes):
    # Ensure boxes are of type float16 (as your model is using float16)
    boxes = tf.cast(boxes, dtype=tf.float16)

    # Extract left, top, right, and bottom coordinates
    l = boxes[:, 0:1]
    t = boxes[:, 1:2]
    r = boxes[:, 2:3]
    b = boxes[:, 3:4]

    # Calculate center coordinates (cx, cy)
    cx = (l + r) / 2
    cy = (t + b) / 2

    # Concatenate cx and cy to get the centers
    centers = tf.concat([cx, cy], axis=1)
    
    return centers


