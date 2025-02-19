from constants import MODEL_WIDTH, MODEL_HEIGHT, MARKER_DIAMETER
import tensorflow as tf

def get_input(video_ref, keypoints=None, padding_ratio=12):
    video_width = video_ref.video_width
    video_height = video_ref.video_height
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

    # Image processing part
    image = tf.image.decode_image(video_ref.read())  # Assuming video_ref is an image reader
    
    # Cropping
    image = image[roi[1]:roi[3], roi[0]:roi[2], :]
    
    # Resizing
    height, width, _ = image.shape
    ratio = height / width
    desired_ratio = MODEL_HEIGHT / MODEL_WIDTH
    resize_height = MODEL_HEIGHT
    resize_width = MODEL_WIDTH
    if ratio > desired_ratio:
        resize_width = int(MODEL_HEIGHT / ratio)
    else:
        resize_height = int(MODEL_WIDTH * ratio)
    
    image = tf.image.resize(image, [resize_height, resize_width])
    
    # Padding
    dx = MODEL_WIDTH - image.shape[1]
    dy = MODEL_HEIGHT - image.shape[0]
    pad_right = dx // 2
    pad_left = dx - pad_right
    pad_bottom = dy // 2
    pad_top = dy - pad_bottom
    padding = [pad_left, pad_right, pad_top, pad_bottom]
    
    image = tf.image.resize_with_crop_or_pad(image, MODEL_HEIGHT, MODEL_WIDTH)
    
    # Normalize the image
    image = image / 255.0

    # Expand dimensions for batch
    image4d = tf.expand_dims(image, axis=0)
    
    return image4d, width, height, padding, roi

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


def get_marker_xy(xy, height, width):
    sx = width / MODEL_WIDTH
    sy = height / MODEL_HEIGHT
    marker_xy = [sx * xy[0], sy * xy[1] - height - MARKER_DIAMETER]
    return marker_xy


def get_centers(boxes):
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

