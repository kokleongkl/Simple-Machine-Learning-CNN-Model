import tensorflow as tf 
import numpy as np
model = tf.keras.models.load_model('models/final_model.h5')

test_image = tf.keras.preprocessing.image.load_img('dataset/test_set/Pulin/20190602_225922_001.jpg',target_size=(128,128))
test_image = tf.keras.preprocessing.image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)
#test_image = test_image.reshape(64, 64)


result = model.predict(test_image,batch_size=1)
print(result)