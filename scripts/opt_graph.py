import argparse
import struct
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib
from tensorflow.tools.graph_transforms import TransformGraph

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='Path to frozen .pb graph')
parser.add_argument('--output', help='Path to ourput optimized .pb graph')
args = parser.parse_args()

pb_file = args.input
graph_def = tf.compat.v1.GraphDef()

try:
    with tf.io.gfile.GFile(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())
except:
    with tf.gfile.FastGFile(pb_file, 'rb') as f:
        graph_def.ParseFromString(f.read())

# Find a node with mean values
for node in graph_def.node:
    if node.name == 'ExpandDims':
        for inp in graph_def.node:
            if inp.name == node.input[0]:
                means = struct.unpack('fff', inp.attr['value'].tensor.tensor_content)
                means = [m * 255 for m in means]
                print('Subtract these mean values for validation:', means)
                break
        break

graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ['image_arrays'], ['detections'], tf.uint8.as_datatype_enum)
graph_def = TransformGraph(graph_def, ['image_arrays'], ['detections'], ['fold_constants'])

with tf.gfile.FastGFile(args.output, 'wb') as f:
   f.write(graph_def.SerializeToString())
