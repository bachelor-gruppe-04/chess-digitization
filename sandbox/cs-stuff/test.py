import cv2
import onnxruntime as ort
import matplotlib.pyplot as plt

# Last inn din egen modell
session = ort.InferenceSession('C:/Users/chris/OneDrive/Skrivebord/Skole/Bachelor/models/480M_leyolo_pieces.onnx')

def preprocess_frame(frame):
    processed_frame = cv2.resize(frame, (480, 288))  # Juster dimensjonene til 288x480
    processed_frame = processed_frame / 255.0  # normalisering
    processed_frame = processed_frame.transpose(2, 0, 1)  # Bytt om dimensjonene for ONNX
    return processed_frame.reshape(1, 3, 288, 480).astype('float16')  # tilpass dimensjonene og endre til float16

def analyse_frame(frame):
    processed_frame = preprocess_frame(frame)
    input_name = session.get_inputs()[0].name
    predictions = session.run(None, {input_name: processed_frame})
    interpret_predictions(predictions)
    return frame

def interpret_predictions(predictions):
    print(predictions)

# Initialiser kamera
cap = cv2.VideoCapture(0)

fig, ax = plt.subplots()
im = ax.imshow(cv2.cvtColor(cap.read()[1], cv2.COLOR_BGR2RGB))

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = analyse_frame(frame)
    im.set_data(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.01)

cap.release()
plt.close()
