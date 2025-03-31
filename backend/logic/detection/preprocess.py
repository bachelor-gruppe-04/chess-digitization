import cv2
import numpy as np
from constants import MODEL_WIDTH, MODEL_HEIGHT

def preprocess_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess the input image to make it compatible with the detection model.

    This function resizes the input image to match the model's expected input size,
    normalizes pixel values, and converts the image to the correct format for the model.
    The pixel values are normalized to the range [0, 1] and the image is cast to float16.

    Args:
        image (numpy.ndarray): The input image to preprocess. The image is in the format (H, W, C) where:
                        - H and W are the height and width of the image respectively.
                        - C represents the number of channels (3 for RGB).

    Returns:
        numpy.ndarray: Preprocessed image ready for inference (with batch dimension and normalized).
        Image is in the format CHW.
    """

    image = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))  # Resize image
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.transpose(2, 0, 1)  # Convert HWC to CHW
    return image[np.newaxis, ...].astype(np.float16)  # Add batch dimension and convert to float16

    #After adding a batch dimension the image is in the shape (1,C,H,W)
    # We need to add a batch dimension to the image to match the input shape expected by the model
    #This you can confirm in netron.app by loading the model and checking the input shape.

    #We convert it to float16 to match the model's input data type. Also float16 saves memory
    #and speeds up computation during inference. Inference is the process of using a trained model to 
    #make predictions on new data.

    #One other thing, normalizing the pixel values helps improve model performance by
    #Ensuring consistent input range, as many models are trained with inputs in the [0, 1] range.