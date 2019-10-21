import tensorflow as tf
saved_model_dir = 'models_able_to use/Best-weights-my_model-165-0.0063-0.9984.h5'
converter = tf.lite.TFLiteConverter.from_keras_model_file(
    saved_model_dir)
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
tflite_quant_model = converter.convert()
file = open("model.tflite", "wb")
file.write(tflite_quant_model)
