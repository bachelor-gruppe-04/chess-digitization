from ultralytics import YOLO
import cv2
from cv2 import imread
import matplotlib.pyplot as plt


model = YOLO.from_pretrained("lhollard/leyolo-medium")
#results = model.val(data="ultralytics/cfg/datasets/coco8.yaml")


def imwrite(img, path):
    if cv2.imwrite(path, img):
        print('Image saved to "{}"'.format(path))
    else:
        print('Failed to save image to "{}"'.format(path))

def imshow(img, figsize=(10, 10), is_bgr=True):
    if is_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=figsize)
    plt.imshow(img, cmap='gray')
    plt.show()

    #load model 
model = YOLO("weights/LeYOLOMedium.pt")

results = model("ultralytics/data/images/cat.jpg")
img = results[0].plot()
imshow(img)