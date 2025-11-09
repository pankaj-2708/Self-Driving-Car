import onnxruntime as ort
import tensorflow as tf
import tf2onnx
from custom_objects import mean_iou,dice_loss,tpr,fpr,dice_coefficient
print(ort.get_device())



model = tf.keras.models.load_model("./models/sterring_angle.h5", custom_objects={'mae': 'mean_absolute_error'},compile=False)

input_shape = model.input_shape[1:]   # removes batch dim (None, H, W, C) -> (H, W, C)

# Create new functional wrapper
inp = tf.keras.Input(shape=input_shape, name="input")
out = model(inp)
wrapped_model = tf.keras.Model(inputs=inp, outputs=out)

# Define TensorSpec (batch=1)
spec = (tf.TensorSpec((1, *input_shape), tf.float32, name="input"),)

tf2onnx.convert.from_keras(wrapped_model, input_signature=spec, opset=17,output_path="models/sterring_angle.onnx")


# lane detection model conversion
lane_model = tf.keras.models.load_model("./models/lane_detection.h5", custom_objects={
                       "dice_loss": dice_loss,
                       "mean_iou": mean_iou,
                       "dice_coefficient": dice_coefficient,
                       "tpr": tpr,
                       "fpr": fpr
                   },compile=False)
input_shape_lane = lane_model.input_shape[1:]   # removes batch dim (None, H, W, C)
# Create new functional wrapper
inp_lane = tf.keras.Input(shape=input_shape_lane, name="input")
out_lane = lane_model(inp_lane)
wrapped_model_lane = tf.keras.Model(inputs=inp_lane, outputs=out_lane)
# Define TensorSpec (batch=1)
spec_lane = (tf.TensorSpec((1, *input_shape_lane), tf.float32, name="input"),)
tf2onnx.convert.from_keras(wrapped_model_lane, input_signature=spec_lane, opset=17,output_path="models/lane_detection.onnx")
