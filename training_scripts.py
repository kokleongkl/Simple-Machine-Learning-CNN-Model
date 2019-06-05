import tensorflow as tf


def learning_rate(epoch):
        if epoch < 50:
                lr = 0.001
        elif epoch < 150:
                lr = 0.0001
        else:
                lr = 0.00001
        return lr
callbacks = tf.keras.callbacks
learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate,verbose=1)

#filepath of logs
filepath_logs = 'log/models_new_train.csv'
csv_log=callbacks.CSVLogger(filepath_logs, separator=',', append=False)
#early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

#filepath for models
filepath="models/Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.h5"
checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [csv_log,checkpoint, learning_rate_scheduler]


#init CNN
classifier = tf.keras.models.Sequential()

#Step 1 Convolution
classifier.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))

#Step 2 Pooling
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

#Adding 2nd Convolution Layer
classifier.add(tf.keras.layers.Conv2D(32,(3,3),input_shape=(64,64,3),activation='relu'))
classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))


#Step 3 Flattening
classifier.add(tf.keras.layers.Flatten())
classifier.add(tf.keras.layers.Dropout(0.2))

#Step 4 Full Connection
classifier.add(tf.keras.layers.Dense(units=128,activation='relu'))
classifier.add(tf.keras.layers.Dropout(0.2))
classifier.add(tf.keras.layers.Dense(units=3,activation='softmax'))


classifier.summary()

#Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            shuffle=True,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 4666,
                         epochs = 5,
                         validation_data = test_set,
                         callbacks = callbacks_list,
                         validation_steps = 933)