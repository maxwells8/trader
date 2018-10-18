import tensorflow as tf
# from tensorflow.contrib.cudnn_rnn import CudnnLSTM

with tf.gfile.GFile('save/best_action.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="prefix")

print([op.name for op in graph.get_operations()])
