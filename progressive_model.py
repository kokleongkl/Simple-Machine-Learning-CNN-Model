import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense, Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
#init check points
filepath="models/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

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
batch_size = 24

# Our training data will use a wide assortment of transformations to try
# and squeeze as much variety as possible out of our image corpus.
# However, for the validation data, we'll apply just one transformation,
# rescaling, because we want our validation set to reflect "real world"
# performance.
#
# Also note that we are using an 80/20 train/validation split.
train_datagen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1/255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
        )
test_datagen = ImageDataGenerator(rescale = 1/255)

training_set = train_datagen.flow_from_directory('dataset/new_training_set_96',
                                                 target_size = (192, 192),
                                                 batch_size = batch_size,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/new_test_set_96',
                                            target_size = (192, 192),
                                            batch_size = batch_size,
                                            shuffle=True,
                                            class_mode = 'categorical')
                                        



# I found that a batch size of 128 offers the best trade-off between
# model training time and batch volatility.

# 2: define the model 
# For the purposes of this article I based the core of my model on VGG16,
# a pretrained CNN architecture somewhat on the simpler side. This version
# of VGG16 is one trained on the famed ImageNet (http://www.image-net.org/)
# which includes some fruits in its list of classes, so performance should
# be decent. I add a new top layer consisting of a large-ish fully 
# connected layer with moderate regularization in the form of dropout.
# There are 12 output classes, so the output layer has 12 nodes.
#

prior = keras.applications.VGG16(
    include_top=False, 
    weights='imagenet',
    input_shape=(192, 192, 3)
)

model = Sequential()
model.add(prior)
model.add(Flatten())
model.add(Dense(128, activation='relu', name='Dense_Intermediate'))
model.add(Dropout(0.4,name='Dropout_Regularization'))
model.add(Dense(3, activation='softmax', name='Output'))


# Freeze the VGG16 model, e.g. do not train any of its weights.
# We will just use it as-is.
for cnn_block_layer in model.layers[0].layers:
    cnn_block_layer.trainable = False
model.layers[0].trainable = False


# Compile the model. I found that RMSprop with the default learning
# weight worked fine.
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)



# Step 4: fit the model.
#
# Finally we fit the model. I use two callbacks here: EarlyStopping,
# which stops the model short of its full 20 epochs if validation 
# performance consistently gets worse; and ReduceLROnPlateau, which 
# reduces the learning rate 10x at a time when it detects model 
# performance is no longer improving between epochs.
#

# Recall that our dataset is highly imbalanced. We deal with this
# problem by generating class weights and passing them to the model
# at training time. The model will use the class weights to adjust
# how it trains so that each class is considered equally important to
# get right, even if the actual distribution of images is highly 
# variable.

import os
labels_count = dict()
for img_class in [ic for ic in os.listdir('dataset/new_training_set_96') if ic[0] != '.']:
    labels_count[img_class] = len(os.listdir('dataset/new_training_set_96/' + img_class))
print(labels_count)
total_count = sum(labels_count.values())
class_weights = {cls: total_count / count for cls, count in 
                 enumerate(labels_count.values())}

print(class_weights)

model.fit_generator(
    training_set,
    steps_per_epoch=12000,
    epochs=15,
    validation_data=test_set,
    validation_steps=3000,
    class_weight=class_weights,
    callbacks=[
        EarlyStopping(patience=3, restore_best_weights=True),
        checkpoint,
        ReduceLROnPlateau(patience=2)
    ]
)

training_set.class_indices


