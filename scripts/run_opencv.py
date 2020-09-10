import argparse
import cv2 as cv

parser = argparse.ArgumentParser()
parser.add_argument('--model')
parser.add_argument('--pbtxt')
parser.add_argument('--width', type=int)
parser.add_argument('--height', type=int)
args = parser.parse_args()

net = cv.dnn_DetectionModel(args.model, args.pbtxt)
net.setInputSize(args.width, args.height)
net.setInputScale(1.0 / 255)
net.setInputMean((123.675, 116.28, 103.53))

frame = cv.imread('images/example.jpg')

classes, confidences, boxes = net.detect(frame, confThreshold=0.5, nmsThreshold=0.4)
for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
    cv.rectangle(frame, box, color=(255, 0, 0), thickness=3)
    cv.rectangle(frame, box, color=(0, 255, 0), thickness=1)

# Uncomment for visualization
# cv.imshow('out', frame)
# cv.waitKey()
