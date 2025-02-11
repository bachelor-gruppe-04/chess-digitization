import cv2
import numpy as np
import onnxruntime as ort

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





def predict_corners(image, corner_ort_session, target_width=480, target_height=288):
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
    corners = predictions[0]  
    return corners

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