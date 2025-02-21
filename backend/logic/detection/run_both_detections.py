import cv2
import numpy as np
import onnxruntime as ort
from piece_detection import predict_pieces, scale_boxes, apply_nms, visualize_boxes_and_labels
from corners_detection import predict_corners, visualize_corners


class_names = {
    0: 'black-bishop', 1: 'black-king', 2: 'black-knight', 3: 'black-pawn', 
    4: 'black-queen', 5: 'black-rook', 6: 'white-bishop', 7: 'white-king', 
    8: 'white-knight', 9: 'white-pawn', 10: 'white-queen', 11: 'white-rook'
}

piece_model_path = "src/logic/models/480M_leyolo_pieces.onnx"
piece_ort_session = ort.InferenceSession(piece_model_path)

corner_model_path = "src/logic/models/480L_leyolo_xcorners.onnx"
corner_ort_session = ort.InferenceSession(corner_model_path)

video_path = 'resources/videos/chessvideo.mp4'
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

output_path = 'resources/videos/output_video_combined.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

frame_counter = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_counter % 10 == 0:
        # Perform piece detection
        xc, yc, w, h, scores, class_indices = predict_pieces(frame, piece_ort_session)
        xc, yc, w, h = scale_boxes(xc, yc, w, h, frame_width, frame_height, 480, 288)
        boxes = np.column_stack((xc, yc, w, h))
        boxes, scores, class_indices = apply_nms(boxes, scores, class_indices)
        xc, yc, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        pieces_frame = visualize_boxes_and_labels(frame.copy(), xc, yc, w, h, class_indices, scores, class_names) 
        corners = predict_corners(frame, corner_ort_session)
        corners_frame = visualize_corners(frame.copy(), corners)

        combined_frame = cv2.addWeighted(pieces_frame, 0.5, corners_frame, 0.5, 0) 

        out.write(combined_frame)

        # Resize frame for display
        resized_frame = cv2.resize(combined_frame, (1280, 720))
        cv2.imshow('Video', resized_frame)

    frame_counter += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()