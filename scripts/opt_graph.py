import argparse
import struct
import tensorflow as tf
from tensorflow.python.tools import optimize_for_inference_lib

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


graph_def = optimize_for_inference_lib.optimize_for_inference(graph_def, ['image_arrays'], ['detections'], tf.uint8.as_datatype_enum)

try:
    with tf.io.gfile.GFile(args.output, 'wb') as f:
        f.write(graph_def.SerializeToString())
except:
    with tf.gfile.FastGFile(args.output, 'wb') as f:
        f.write(graph_def.SerializeToString())
