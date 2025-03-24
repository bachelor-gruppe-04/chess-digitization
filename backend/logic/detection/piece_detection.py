import cv2

from detection_methods import get_input, get_boxes_and_scores, process_boxes_and_scores
from preprocess import preprocess_image

def predict_pieces(frame, ort_session):
    """
    Predict bounding boxes, class indices, and scores using the ONNX model.

    This function processes the input image, runs inference on an ONNX model, 
    and returns the predicted bounding boxes, class indices, and confidence scores.

    Args:
        image (numpy array): Input image for prediction. Expected shape is (height, width, channels).
        ort_session: ONNX Runtime InferenceSession object used to run inference on the ONNX model.

    Returns:
        tuple: A tuple containing the predictions, which typically include:
               - Bounding box coordinates (xc, yc, w, h)
               - Confidence scores for each box
               - Class indices indicating which object each bounding box corresponds to
    """
    # Preprocess the image (resize, normalize, etc.) before feeding it to the model
    preprocessed_image = preprocess_image(frame)

    # Get input and output information from the ONNX model
    model_inputs = ort_session.get_inputs()  # Get the input layers of the model
    model_outputs = ort_session.get_outputs()  # Get the output layers of the model

    # Run inference on the preprocessed image using the ONNX model
    predictions = ort_session.run(
        output_names=[output.name for output in model_outputs],  # Specify the model outputs to collect
        input_feed={model_inputs[0].name: preprocessed_image}  # Feed the preprocessed image to the model
    )

    # Extract the predicted values (bounding boxes, class indices, and scores) from the output
    pieces_predictions = predictions[0]

    return pieces_predictions



async def run_pieces_model(frame, pieces_model_ref):
    """
    Processes a video reference using a given pieces detection model to predict chess pieces in the video.
    
    Parameters:
    - frame: A reference to the video data.
    - pieces_model_ref: A reference to the pieces detection model used to make predictions on the video frame.
    
    Returns:
    - pieces: A list of detected chess pieces with their associated bounding boxes and class labels.
    """
    
    frame_height, frame_width, _ = frame.shape

    # Prepare the input data for the pieces detection model
    image4d, width, height, padding, roi = get_input(frame)

    # Predict the pieces. This includes bounding boxes and which chess-piece
    pieces_prediction = predict_pieces(frame, pieces_model_ref)

    # Extract the bounding boxes and scores for the predicted pieces
    # This function processes the raw prediction to provide:
    # - boxes: Bounding boxes around the detected pieces
    # - scores: Confidence scores for each predicted bounding box
    boxes, scores = get_boxes_and_scores(pieces_prediction, width, height, frame_width, frame_height, padding, roi)

    # Process the boxes and scores using non-max suppression and other techniques
    # This will clean up the results by removing redundant boxes
    # Basically returns the best 16 pieces for white and best 16 pieces for black. 
    # Format for each entry is (x,y,pieceTypeIndex)
    pieces = process_boxes_and_scores(boxes, scores)

    # Cleanup: Delete intermediate variables to free up memory
    del pieces_prediction
    del image4d  
    del boxes  

    return pieces



def scale_boxes(xc, yc, w, h, orig_width, orig_height, target_width, target_height):
    """
    Scale the bounding box coordinates back to the original image size.

    Args:
        xc, yc, w, h (numpy array): The bounding box coordinates.
        orig_width, orig_height (int): Original image dimensions.
        target_width, target_height (int): Target image dimensions used for resizing.

    Returns:
        tuple: Scaled bounding box coordinates.
    """
    xc = xc * (orig_width / target_width)
    yc = yc * (orig_height / target_height)
    w = w * (orig_width / target_width)
    h = h * (orig_height / target_height)
    return xc, yc, w, h



def apply_nms(boxes, scores, class_indices, nms_threshold=0.5):
    """
    Apply Non-Maximum Suppression (NMS) to remove overlapping bounding boxes.

    Args:
        boxes (numpy array): Bounding box coordinates.
        scores (numpy array): Confidence scores for each bounding box.
        class_indices (numpy array): Predicted class indices for each box.
        nms_threshold (float): Threshold to use for NMS.

    Returns:
        tuple: Filtered bounding boxes, scores, and class indices.
    """
    indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), score_threshold=0.0, nms_threshold=nms_threshold)
    indices = indices.flatten() if indices is not None else []
    boxes = boxes[indices]
    scores = scores[indices]
    class_indices = class_indices[indices]
    return boxes, scores, class_indices



def visualize_boxes_and_labels(image, xc, yc, w, h, class_indices, scores, class_names):
    """
    Visualize bounding boxes and labels on the image.

    Args:
        image (numpy array): Image to draw the bounding boxes on.
        xc, yc, width, height (numpy array): Coordinates and size of the bounding boxes.
        class_indices (numpy array): Indices of the predicted classes.
        scores (numpy array): Confidence scores for each bounding box.
        class_names (dict): Dictionary mapping class indices to class names.

    Returns:
        numpy array: Image with bounding boxes drawn.
    """
    for i in range(xc.shape[0]):
        # Calculate the coordinates of the bounding box
        x_min = xc[i] - w[i] / 2
        y_min = yc[i] - h[i] / 2
        x_max = xc[i] + w[i] / 2
        y_max = yc[i] + h[i] / 2

        # Draw bounding box
        cv2.rectangle(image, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (255, 0, 0), 2)

        # Get class label and score
        class_name = class_names[class_indices[i]]
        score = scores[i]

        # Add label to the bounding box
        label = f"{class_name}: {score:.2f}"
        cv2.putText(image, label, (int(x_min), int(y_min) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 50, 50), 2)

    return image