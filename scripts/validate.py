import argparse
import numpy as np
import cv2 as cv
from openvino.inference_engine import IECore

parser = argparse.ArgumentParser()
parser.add_argument('--version', required=True)
parser.add_argument('--width', type=int)
parser.add_argument('--height', type=int)
args = parser.parse_args()

img = cv.imread('images/example.jpg')
inp = cv.resize(img, (args.width, args.height))

#
# Run TensorFlow
#
import tensorflow as tf

pb_file = 'automl/efficientdet/savedmodeldir/efficientdet-{}_frozen.pb'.format(args.version)
graph_def = tf.compat.v1.GraphDef()

try:
    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
except:
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    tfOut = sess.run(sess.graph.get_tensor_by_name('detections:0'),
                     feed_dict={'image_arrays:0': inp.reshape(1, args.height, args.width, 3)})

#
# Run OpenVINO
#
inp = inp.astype(np.float32)
inp[:,:,0] -= 123.675
inp[:,:,1] -= 116.28
inp[:,:,2] -= 103.53
inp = inp * 1.0 / 255
inp = inp.transpose(2, 0, 1).reshape(1, 3, args.height, args.width)


ie = IECore()
net = ie.read_network('efficientdet-{}.xml'.format(args.version),
                      'efficientdet-{}.bin'.format(args.version))
exec_net = ie.load_network(net, 'CPU')
ieOut = exec_net.infer({'image_arrays': inp})
ieOut = next(iter(ieOut.values()))


# Render detections
print('\nTensorFlow predictions')
tfOut = tfOut.reshape(-1, 7)
for detection in tfOut:
    # Normalize coordinates
    detection[1] /= args.height
    detection[2] /= args.width
    detection[3] /= args.height
    detection[4] /= args.width

    conf = detection[5]
    if conf < 0.5:
        continue
    ymin = int(detection[1] * float(img.shape[0]))
    xmin = int(detection[2] * float(img.shape[1]))
    ymax = int(detection[3] * float(img.shape[0]))
    xmax = int(detection[4] * float(img.shape[1]))
    cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 127, 255), thickness=3)
    print(conf, xmin, ymin, xmax, ymax)


print('\nOpenVINO predictions')
ieOut = ieOut.reshape(-1, 7)
for detection in ieOut:
    conf = detection[2]
    if conf < 0.5:
        continue

    xmin = int(img.shape[1] * detection[3])
    ymin = int(img.shape[0] * detection[4])
    xmax = int(img.shape[1] * detection[5])
    ymax = int(img.shape[0] * detection[6])
    cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), thickness=1)
    print(conf, xmin, ymin, xmax, ymax)


# Uncomment for visualization
# cv.imshow('res', img)
# cv.waitKey()


# Test source code: https://github.com/opencv/opencv/blob/master/modules/dnn/misc/python/test/test_dnn.py
def inter_area(box1, box2):
    x_min, x_max = max(box1[0], box2[0]), min(box1[2], box2[2])
    y_min, y_max = max(box1[1], box2[1]), min(box1[3], box2[3])
    return (x_max - x_min) * (y_max - y_min)

def area(box):
    return (box[2] - box[0]) * (box[3] - box[1])

def box2str(box):
    left, top = box[0], box[1]
    width, height = box[2] - left, box[3] - top
    return '[%f x %f from (%f, %f)]' % (width, height, left, top)

def normAssertDetections(refClassIds, refScores, refBoxes, testClassIds, testScores, testBoxes,
                         confThreshold=0.5, scores_diff=1e-5, boxes_iou_diff=1e-4):
    matchedRefBoxes = [False] * len(refBoxes)
    errMsg = ''
    for i in range(len(testBoxes)):
        testScore = testScores[i]
        if testScore < confThreshold:
            continue

        testClassId, testBox = testClassIds[i], testBoxes[i]
        matched = False
        for j in range(len(refBoxes)):
            if (not matchedRefBoxes[j]) and testClassId == refClassIds[j] and \
               abs(testScore - refScores[j]) < scores_diff:
                interArea = inter_area(testBox, refBoxes[j])
                iou = interArea / (area(testBox) + area(refBoxes[j]) - interArea)
                if abs(iou - 1.0) < boxes_iou_diff:
                    matched = True
                    matchedRefBoxes[j] = True
        if not matched:
            errMsg += '\nUnmatched prediction: class %d score %f box %s' % (testClassId, testScore, box2str(testBox))

    for i in range(len(refBoxes)):
        if (not matchedRefBoxes[i]) and refScores[i] > confThreshold:
            errMsg += '\nUnmatched reference: class %d score %f box %s' % (refClassIds[i], refScores[i], box2str(refBoxes[i]))
    if errMsg:
        print(errMsg)
        exit(1)


tfOut = tfOut[:,[0, 2, 1, 4, 3, 5, 6]]  # yxYX -> xyXY

normAssertDetections(refClassIds=tfOut[:,6], refScores=tfOut[:,5], refBoxes=tfOut[:,1:5],
                     testClassIds=ieOut[:,1] + 1, testScores=ieOut[:,2], testBoxes=ieOut[:,3:7])
