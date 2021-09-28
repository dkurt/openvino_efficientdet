import argparse
import struct
import numpy as np
import cv2 as cv
from openvino.inference_engine import IECore

parser = argparse.ArgumentParser()
parser.add_argument('--version', required=True)
parser.add_argument('--width', type=int, required=True)
parser.add_argument('--height', type=int, required=True)
parser.add_argument('--fp16', dest='fp16', default=False, action='store_true')
args = parser.parse_args()

img = cv.imread('images/example.jpg')
img_h, img_w = img.shape[0], img.shape[1]

conf_threshold = 0.5

if args.fp16:
    print("Evaluating accuracy for FP16")
else:
    print("Evaluating accuracy for FP32")
#
# Run TensorFlow
#
import tensorflow as tf

pb_file = 'automl/efficientdet/savedmodeldir-{}/efficientdet-{}_frozen.pb'.format(args.version, args.version)
graph_def = tf.compat.v1.GraphDef()

try:
    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
except:
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())

with tf.compat.v1.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    for node in graph_def.node:
        if node.name == 'Const_3':
            means = struct.unpack('fff', node.attr['value'].tensor.tensor_content)
            print('Mean values are', [m * 255 for m in means])

    tfOut = sess.run(sess.graph.get_tensor_by_name('detections:0'),
                     feed_dict={'image_arrays:0': np.expand_dims(img, axis=0)})

#
# Run OpenVINO
#
h, w = img.shape[0:2]
# 1. Resize and keep aspect ratio
assert(w >= h)
h = int(h / w * args.height)
inp = cv.resize(img.astype(np.float32), (args.width, h))  # It's important to perform resize for fp32
# 2. Zero padding to the bottom
inp = np.pad(inp, ((0, args.height - h), (0, 0), (0, 0)), 'constant')
inp = np.expand_dims(inp.transpose(2, 0, 1), axis=0)
# 3. Add means (to imitate zero padding after internal mean subtraction)
inp[0,0,h:,:] += 123.67500364780426
inp[0,1,h:,:] += 116.28000006079674
inp[0,2,h:,:] += 103.52999702095985


ie = IECore()
net = ie.read_network('efficientdet-{}.xml'.format(args.version),
                      'efficientdet-{}.bin'.format(args.version))
exec_net = ie.load_network(net, 'CPU')
inp_name = next(iter(exec_net.input_info.keys()))
ieOut = exec_net.infer({inp_name: inp})
ieOut = next(iter(ieOut.values()))


# Render detections
print('\nTensorFlow predictions')
tfOut = tfOut.reshape(-1, 7)
for detection in tfOut:
    conf = detection[5]
    if conf < conf_threshold:
        continue
    ymin, xmin, ymax, xmax = [int(v) for v in detection[1:5]]
    cv.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 127, 255), thickness=3)
    print(conf, xmin, ymin, xmax, ymax)


print('\nOpenVINO predictions')
ieOut = ieOut.reshape(-1, 7)
for detection in ieOut:
    conf = detection[2]
    if conf < conf_threshold:
        continue

    xmin = int(detection[3] * img_w)
    xmax = int(detection[5] * img_w)
    # Recalculate y coordinates excluding paddings
    ymin = int(img_h * detection[4] * (args.height / h))
    ymax = int(img_h * detection[6] * (args.height / h))

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
                         confThreshold=conf_threshold, scores_diff=1e-5, boxes_iou_diff=1e-4):
    matchedRefBoxes = [False] * len(refBoxes)
    errMsg = ''
    for i in range(len(testBoxes)):
        testScore = testScores[i]
        if testScore < confThreshold:
            continue

        testClassId, testBox = testClassIds[i], testBoxes[i]
        matched = False
        top_iou = 0
        for j in range(len(refBoxes)):
            if (not matchedRefBoxes[j]) and testClassId == refClassIds[j] and \
               abs(testScore - refScores[j]) < scores_diff:
                interArea = inter_area(testBox, refBoxes[j])
                iou = interArea / (area(testBox) + area(refBoxes[j]) - interArea)
                top_iou = max(iou, top_iou)
                if iou - 1.0 < boxes_iou_diff:
                    matched = True
                    matchedRefBoxes[j] = True
        if not matched:
            errMsg += '\nUnmatched prediction: class %d score %f box %s' % (testClassId, testScore, box2str(testBox))
            errMsg += ' (highest IoU: {})'.format(top_iou)

    for i in range(len(refBoxes)):
        if (not matchedRefBoxes[i]) and refScores[i] > confThreshold:
            errMsg += '\nUnmatched reference: class %d score %f box %s' % (refClassIds[i], refScores[i], box2str(refBoxes[i]))
    if errMsg:
        print(errMsg)
        exit(1)


tfOut = tfOut[:,[0, 2, 1, 4, 3, 5, 6]]  # yxYX -> xyXY

if args.fp16 is False:
    normAssertDetections(refClassIds=tfOut[:,6], refScores=tfOut[:,5], refBoxes=tfOut[:,1:5],
                        testClassIds=ieOut[:,1] + 1, testScores=ieOut[:,2], testBoxes=ieOut[:,3:7])
else:
    normAssertDetections(refClassIds=tfOut[:,6], refScores=tfOut[:,5], refBoxes=tfOut[:,1:5],
                        testClassIds=ieOut[:,1] + 1, testScores=ieOut[:,2], testBoxes=ieOut[:,3:7],
                        confThreshold=conf_threshold, scores_diff=0.05)
