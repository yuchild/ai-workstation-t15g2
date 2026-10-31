import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")
print(f"TensorFlow: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")
print(f"GPU devices: {gpus}")

assert gpus, "TensorFlow cannot access the NVIDIA GPU"

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

with tf.device("/GPU:0"):
    x = tf.random.normal((4096, 4096))
    y = tf.matmul(x, x)

print(f"Compute device: {y.device}")
print(f"Result shape: {y.shape}")
