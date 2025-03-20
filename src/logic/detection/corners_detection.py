from constants import MODEL_WIDTH, MODEL_HEIGHT

import cv2
import numpy as np
import tensorflow as tf
from maths import clamp
from quad_transformation import get_quads, score_quad, perspective_transform
from detection_methods import get_centers_of_bbox, get_input, get_boxes_and_scores

def preprocess_corner_image(image, target_width, target_height):
    """
    Preprocess the input image for corner detection model.

    Args:
        image (numpy.ndarray): The input image to preprocess.
        target_width (int): The target width for resizing the image.
        target_height (int): The target height for resizing the image.

    Returns:
        numpy.ndarray: Preprocessed image ready for inference (with batch dimension and normalized).
    """
    image = cv2.resize(image, (target_width, target_height))  # Resize image
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.transpose(2, 0, 1)  # Convert Height, Width, Channels (HWC) to Channels, Height, Width format (CHW)
    return image[np.newaxis, ...].astype(np.float16)  # Add batch dimension and convert to float16

    #After adding a batch dimension the image is in the shape (1,C,H,W)
    #In CHW the C or channels represents the color channels in the image, in our case RGB (3)
    #And in grayscale images there is only 1 channel

    #We need to add a batch dimension to the image to match the input shape expected by the model
    #The model expects the input shape to be (1, 3, 288, 480) for the corner detection model. 
    #This you can confrim in netron.app by loading the model and checking the input shape.

    #We convert it to float16 to match the model's input data type. Also float16 saves memory
    #and speeds up computation during inference. Inference is the process of using a trained model to 
    #make predictions on new data.

    #One other thing, normalizing the pixel values helps improve model performance by
    #Ensuring consistent input range, as many models are trained with inputs in the [0, 1] range.



def get_prediction_corners(image, corner_ort_session, target_width=480, target_height=288):
    """
    Perform corner detection on the input image.

    Args:
        image (numpy.ndarray): The input image to detect corners.
        corner_ort_session: The ONNX Runtime InferenceSession object for the corner detection model.
        target_width (int, optional): The width to resize the image. Default is 480.
        target_height (int, optional): The height to resize the image. Default is 288.
        confidence_threshold (float, optional): The threshold for filtering low-confidence corner predictions. Default is 0.2.

    Returns:
        numpy.ndarray: Array of predicted corner points (coordinates and confidence scores).
    """
    preprocessed_image = preprocess_corner_image(image, target_width, target_height)

    # Run inference for corner detection
    model_inputs = corner_ort_session.get_inputs()
    model_outputs = corner_ort_session.get_outputs()
    predictions = corner_ort_session.run(
        output_names=[output.name for output in model_outputs],
        input_feed={model_inputs[0].name: preprocessed_image}
    )

    #After doing the inference we get a prediction of where the corners are in this image. The format
    #of the prediction is on the form float16[1,5,2835] where 1 is batch size, 5 represents the coordiantes of
    #the corners (4) and (1) for confidence score. Lastly the 2835 is the number of n_anchors or anchor boxes. Anchor
    #boxes are pre-defined boxes that the model uses to predict the corners. The image is a grid of 2835 tiny boxes 
    #where the model uses these to predict the corners
    corner_predictions = predictions[0]  

    return corner_predictions


def process_boxes_and_scores(boxes, scores):
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

    # Convert the result to a NumPy array (for further handling outside TensorFlow)
    res_array = res.numpy()
    
    return res_array


async def run_xcorners_model(video_ref, corners_model_ref, pieces):
    video_height, video_width, _ = video_ref.shape

    keypoints = [[x[0], x[1]] for x in pieces]

    image4d, width, height, padding, roi = get_input(video_ref, keypoints)


    corner_predictions = get_prediction_corners(video_ref, corners_model_ref)

    boxes,scores = get_boxes_and_scores(corner_predictions, width, height, video_width, video_height, padding, roi)

    del corner_predictions
    del image4d

    x_corners = process_boxes_and_scores(boxes,scores)



    x_corners = [[x[0], x[1]] for x in x_corners]

    return x_corners



def find_corners_from_xcorners(x_corners):
    quads = get_quads(x_corners)
    
    if len(quads) == 0:
        return None
    
    best_score = None
    best_m = None
    best_offset = None
    
    # Assume score_quad returns score, M, offset
    best_score, best_m, best_offset = score_quad(quads[0], x_corners)
    
    for quad in quads[1:]:
        score, m, offset = score_quad(quad, x_corners)
        if score > best_score:
            best_score = score
            best_m = m
            best_offset = offset
    
    # Inverse matrix calculation
    inv_m = np.linalg.inv(best_m)
    
    # Define warped corners based on offset
    warped_corners = [
        [best_offset[0] - 1, best_offset[1] - 1],
        [best_offset[0] - 1, best_offset[1] + 7],
        [best_offset[0] + 7, best_offset[1] + 7],
        [best_offset[0] + 7, best_offset[1] - 1]
    ]
    
    # Apply perspective transform
    corners = perspective_transform(warped_corners, inv_m)
    
    # Clip corners
    for i in range(4):
        corners[i][0] = clamp(corners[i][0], 0, MODEL_WIDTH)
        corners[i][1] = clamp(corners[i][1], 0, MODEL_HEIGHT)
            
    return corners


def get_center(points):
    center = [sum(x[0] for x in points), sum(x[1] for x in points)]
    center = [x / len(points) for x in center]
    return center

def euclidean(a, b):
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dist = (dx ** 2 + dy ** 2) ** 0.5
    return dist



def calculate_keypoints(black_pieces, white_pieces, corners):
    black_center = get_center(black_pieces)
    white_center = get_center(white_pieces)
    
    best_shift = 0
    best_score = 0
    for shift in range(4):
        cw = [(corners[shift % 4][0] + corners[(shift + 1) % 4][0]) / 2,
              (corners[shift % 4][1] + corners[(shift + 1) % 4][1]) / 2]
        cb = [(corners[(shift + 2) % 4][0] + corners[(shift + 3) % 4][0]) / 2,
              (corners[(shift + 2) % 4][1] + corners[(shift + 3) % 4][1]) / 2]
        score = 1 / (1 + euclidean(white_center, cw) + euclidean(black_center, cb))
        if score > best_score:
            best_score = score
            best_shift = shift

    keypoints = {
        "a1": corners[best_shift % 4],
        "h1": corners[(best_shift + 1) % 4],
        "h8": corners[(best_shift + 2) % 4],
        "a8": corners[(best_shift + 3) % 4]
    }
    return keypoints