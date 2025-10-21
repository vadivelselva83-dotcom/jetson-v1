# file: convert_tftrt.py
import tensorflow as tf
import os

SAVEDMODEL_DIR = "outputs_retinanet/savedmodel"
TRT_DIR = "outputs_retinanet/savedmodel_trt_fp16"
os.makedirs(TRT_DIR, exist_ok=True)

params = tf.experimental.tensorrt.ConversionParams(
    precision_mode="FP16",
    max_workspace_size_bytes=1<<30  # 1GB
)
converter = tf.experimental.tensorrt.Converter(
    input_saved_model_dir=SAVEDMODEL_DIR,
    conversion_params=params
)
converter.convert()
def my_input_fn():
    # provide a few warmup shapes
    import numpy as np
    for _ in range(10):
        yield [np.random.rand(1, 640, 640, 3).astype("float32")]
converter.build(input_fn=my_input_fn)
converter.save(TRT_DIR)
print("TF-TRT model saved to:", TRT_DIR)
