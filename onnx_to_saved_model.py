import onnx
from onnx_tf.backend import prepare

onnx_model = onnx.load("test.onnx")
tf_rep = prepare(onnx_model)
tf_rep.export_graph(".")