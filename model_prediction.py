import tensorflow as tf 
import numpy as np
model = tf.keras.models.load_model('models_able_to use/Best-weights-my_model-019-0.0001-1.0000.h5')

test_image = tf.keras.preprocessing.image.load_img('20190610_103506.jpg',target_size=(192,192))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
#test_image = test_image.reshape(64, 64)


result = model.predict(test_image,batch_size=1)
print(result)