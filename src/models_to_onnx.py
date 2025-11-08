import onnxruntime as ort
import tensorflow as tf
import tf2onnx

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

# with open("models/sterring_angle.onnx", "wb") as f:
#     f.write(onnx_model.SerializeToString())
