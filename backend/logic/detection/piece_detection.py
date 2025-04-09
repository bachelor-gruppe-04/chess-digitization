import cv2

from detection_methods import get_input, get_boxes_and_scores, process_boxes_and_scores

def predict_pieces(image4d, ort_session):
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
    # Get input and output information from the ONNX model
    model_inputs = ort_session.get_inputs()  # Get the input layers of the model
    model_outputs = ort_session.get_outputs()  # Get the output layers of the model

    # Run inference on the preprocessed image using the ONNX model
    predictions = ort_session.run(
        output_names=[output.name for output in model_outputs],  # Specify the model outputs to collect
        input_feed={model_inputs[0].name: image4d}  # Feed the preprocessed image to the model
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
    pieces_prediction = predict_pieces(image4d, pieces_model_ref)

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