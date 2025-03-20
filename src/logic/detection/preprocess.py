import cv2
import numpy as np
from constants import MODEL_WIDTH, MODEL_HEIGHT

def preprocess_image(image):
    print("hheeh")
    print(type(image))
    image = cv2.resize(image, (MODEL_WIDTH, MODEL_HEIGHT))  # Resize image
    image = image / 255.0  # Normalize pixel values to [0, 1]
    image = image.transpose(2, 0, 1)  # Convert Height, Width, Channels (HWC) to Channels, Height, Width format (CHW)
    return image[np.newaxis, ...].astype(np.float16)  # Add batch dimension and convert to float16
