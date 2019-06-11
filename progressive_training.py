import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense,Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau,ModelCheckpoint

#step 1: define the data generators
#
#Data generators are on-the-fly image transformers and are recommended 
# way of providing data to models to Keras. They let you work with 
# on-disk image data too large to fit all at once in-memory. And allow
#you to preprocess the images your model sees with random image
# transformations and standardizations, a key technique for improving model
# performance. To learn more , see https://keras.io/preprocessing/image/.

#our training data will use a wide assortment of transsformations to try
# and squeeze as much variey as possible out of our image corpus.
# However, for the validation data, we'll apply just one transformation,
# rescaling, because we want our validation set to reflect the "real world"
# performance

#spliting data 80/20 training/validation split

filepath="models/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

train_datagen = ImageDataGenerator(
    rotation_range= 40,
    width_shift_range =0.2,
    height_shift_range =0.2,
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


# batch size 128 offers best trade off
batch_size= 128

train_generator = train_datagen.flow_from_directory(
    'dataset/training_set',
    target_size=(48,48),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'training',
)


validation_generator = train_datagen.flow_from_directory(
    'dataset/new_test_set',
    target_size = (48,48),
    batch_size = batch_size,
    class_mode = 'categorical',
    subset = 'validation'
)


# define models

# used VGG16 layer instead of 2DConv
prior =  tf.keras.applications.VGG16(
    include_top =False,
    weights='imagenet',
    input_shape=(48,48,3)
)

model = Sequential()
model.add(prior)
model.add(Flatten())
model.add(Dense(256,activation='relu',name='Dense_Intermediate'))
model.add(Dropout(0.1,name='Dropout_Regularization'))
model.add(Dense(3,activation='sigmoid',name="output"))


#freeze VGG16 model , not to use thw weights to train
for cnn_bloack_layer in model.layers[0].layers:
    cnn_bloack_layer.trainable = False
model.layers[0].trainable = False

#Complie the model
model.compile(
    optimizer=RMSprop(),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)


#fit the model
import os
labels_count = dict()
for img_class in [ic for ic in os.listdir('dataset/new_training_set/') if ic[0]!='.']:
    labels_count[img_class]= len(os.listdir('dataset/new_training_set/'+img_class))


total_count = sum(labels_count.values())
class_weight = {cls:total_count/count for cls,count in enumerate(labels_count.values())}


model.fit_generator(
    train_generator,
    steps_per_epoch=len(train_generator.filenames) // batch_size,
    epochs =20,
    validation_steps = len(train_generator.filenames) //batch_size,
    class_weight =class_weight,
    callbacks = [
        EarlyStopping(patience=3,restore_best_weights=True),
        ReduceLROnPlateau(patience=2),
        checkpoint

    ]
)