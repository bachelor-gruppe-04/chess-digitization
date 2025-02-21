from constants import MODEL_WIDTH, MODEL_HEIGHT
from detect import get_input

import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
from detect import get_centers
from maths import clamp
from quad_transformation import get_quads, score_quad, perspective_transform


corner_model_path = "src/logic/models/480L_leyolo_xcorners.onnx"
corner_ort_session = ort.InferenceSession(corner_model_path)

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

def visualize_corners(image, corners, target_width=480, target_height=288, confidence_threshold=0.2):
    """
    Visualizes corner points on an image.

    Args:
        image (numpy.ndarray): The image on which to visualize corners.
        corners (numpy.ndarray): Predicted corner points (coordinates and confidence scores).
        target_width (int, optional): The width to resize the image. Default is 480.
        target_height (int, optional): The height to resize the image. Default is 288.
        confidence_threshold (float, optional): The threshold for filtering low-confidence corner predictions. Default is 0.2.

    Returns:
        numpy.ndarray: The image with corner points visualized.
    """
    frame_height, frame_width = image.shape[:2]
    for corner in corners.T: 
        x, y, w, h, conf = corner

        if conf < confidence_threshold:
            continue 

        x = int(x * frame_width / target_width)
        y = int(y * frame_height / target_height)

        cv2.circle(image, (x, y), radius=5, color=(0, 255, 0), thickness=-1)
    return image

    #We find the x and y coordinates from the corners we found in predictions[0]. We only
    #want to draw the points with high confidence threshold (conf < confidence_thrsehold). Then multiply by
    #frame_width and frame_height to get the actual coordinates of the corners in the image and

    #use cv2.circle to draw the points

def process_boxes_and_scores(boxes, scores):
    max_scores = tf.reduce_max(scores, axis=1)
    argmax_scores = tf.argmax(scores, axis=1)
    nms = tf.image.non_max_suppression(boxes, max_scores, max_output_size=100, iou_threshold=0.3, score_threshold=0.1)
    
    # Use get_centers function to get the centers from the selected boxes
    centers = get_centers(tf.gather(boxes, nms, axis=0))

    # Gather the class indices of the selected boxes and expand dimensions
    cls = tf.expand_dims(tf.gather(argmax_scores, nms, axis=0), axis=1)

    # Cast cls to float16 (ensure it's compatible with centers)
    cls = tf.cast(cls, dtype=tf.float16)

    # Concatenate the centers with the class indices
    res = tf.concat([centers, cls], axis=1)

    # Convert the result to a NumPy array (for further handling outside TensorFlow)
    res_array = res.numpy()
    
    return res_array

def run_pieces_model(video_ref, pieces_model_ref):
    video_height, video_width, _ = video_ref.shape
    image4d, width, height, padding, roi = get_input(video_ref)

    # Predict using the model
    pieces_preds = pieces_model_ref.predict(image4d)

    # Extract boxes and scores
    boxes_and_scores = get_boxes_and_scores(pieces_preds, width, height, video_width, video_height, padding, roi)

    # Process boxes and scores
    pieces = process_boxes_and_scores(boxes_and_scores['boxes'], boxes_and_scores['scores'])




async def run_xcorners_model(video_frame):
        """
        Runs the xcorners model on a video frame and returns processed xCorners.

        Parameters:
            video_frame (numpy array): The video frame to process.
            xcorners_model (tf.keras.Model): The trained xcorners model.
            pieces (list of lists): Keypoints as [x, y] coordinates.

        Returns:
            list of lists: xCorners as [x_center, y_center] after processing.
        """
        # Convert pieces to keypoints (list of [x, y] coordinates)
        #DO LATER FOR OPTIMIZING WHAT TO READ FROM IMAGE
        # keypoints = [[x[0], x[1]] for x in pieces]

        # Get video dimensions
        video_height, video_width, _ = video_frame.shape

        # # Step 1: Preprocess input (getInput equivalent)
        #DO LATER FOR OPTIMIZING WHAT TO READ FROM IMAGE
        # image4D, width, height = get_input(video_frame, keypoints)

        # Step 2: Run the model prediction
        corner_predictions = get_prediction_corners(video_frame, corner_ort_session)


        # Step 3: Post-process to get boxes and scores
        boxes, scores = get_boxes_and_scores(corner_predictions, 1920, 1080, video_width, video_height)

        # Step 4: Clean up tensors
        tf.keras.backend.clear_session()

        # Step 5: Process boxes and scores
        x_corners = process_boxes_and_scores(boxes, scores)

        # Step 6: Extract the centers
        x_corners = [[x[0], x[1]] for x in x_corners]  # Keep only the centers

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


def get_boxes_and_scores(preds, width, height, video_width, video_height):
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

    # Scale the bounding box coordinates
    l = l * (width / MODEL_WIDTH)
    r = r * (width / MODEL_WIDTH)
    t = t * (height / MODEL_HEIGHT)
    b = b * (height / MODEL_HEIGHT)

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