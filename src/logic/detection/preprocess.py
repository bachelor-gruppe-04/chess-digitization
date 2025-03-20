import cv2
import numpy as np
from constants import MODEL_WIDTH, MODEL_HEIGHT

def preprocess_image(image):
    """
    Preprocess the input image for corner detection model.

    Args:
        image (numpy.ndarray): The input image to preprocess.
        target_width (int): The target width for resizing the image.
        target_height (int): The target height for resizing the image.

    Returns:
        numpy.ndarray: Preprocessed image ready for inference (with batch dimension and normalized).
    """


    image = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))  # Resize image
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