import tensorflow as tf
import numpy as np
model = tf.keras.models.load_model(
    'models/Best-weights-my_model-165-0.0063-0.9984.h5')


test_image = tf.keras.preprocessing.image.load_img(
    '20190617_154028.jpg', target_size=(128, 128))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
#test_image = test_image.reshape(64, 64)


result = model.predict_classes(test_image, batch_size=1)
y_classes = result.argmax(axis=-1)
print(result)
print(y_classes)
