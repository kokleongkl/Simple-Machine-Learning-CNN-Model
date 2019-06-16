import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense, Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint



#
# Step 1: define the data generators.
#
# Data generators are on-the-fly image transformers and are the recommended
# way of providing image data to models in Keras. They let you work with
# on-disk image data too large to fit all at once in-memory. And they allow
# you to preprocess the images your model sees with random image 
# transformations and standardizations, a key technique for improving model
# performance. To learn more, see https://keras.io/preprocessing/image/.
# 


# Our training data will use a wide assortment of transformations to try
# and squeeze as much variety as possible out of our image corpus.
# However, for the validation data, we'll apply just one transformation,
# rescaling, because we want our validation set to reflect "real world"
# performance.
#
# Also note that we are using an 80/20 train/validation split.
train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    rescale=1/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2
)
test_datagen = ImageDataGenerator(
    rescale=1/255,
)
# I found that a batch size of 128 offers the best trade-off between
# model training time and batch volatility.
batch_size = 128

train_generator = train_datagen.flow_from_directory(
    'dataset/new_training_set_96',
    target_size=(96, 96),
    batch_size=batch_size,
    class_mode='categorical',
)

validation_generator = train_datagen.flow_from_directory(
    'dataset/new_test_set_96',
    target_size=(96, 96),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)


filepath="models/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')


model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(96, 96, 3), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(3, 3), input_shape=(96, 96, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

prior = tf.keras.models.load_model('models/Best-weights-my_model-010-0.1165-0.9596.h5')

for layer in prior.layers[0].layers[2:]:
    model.add(layer)


for layer in prior.layers[1:]:
    model.add(layer)

for layer in prior.layers[-4:]:
    layer.trainable = False


model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#fit the model
import os
labels_count = dict()
for img_class in [ic for ic in os.listdir('dataset/new_training_set_96/') if ic[0]!='.']:
    labels_count[img_class]= len(os.listdir('dataset/new_training_set_96/'+img_class))


total_count = sum(labels_count.values())
class_weight = {cls:total_count/count for cls,count in enumerate(labels_count.values())}


model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs =5,
    validation_steps = len(train_generator.filenames) //batch_size,
    class_weight =class_weight,
    callbacks = [
        #EarlyStopping(patience=3,restore_best_weights=True),
        ReduceLROnPlateau(patience=2),
        checkpoint

    ]
)