import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def learning_rate(epoch):
        if epoch < 50:
                lr = 0.001
        elif epoch < 150:
                lr = 0.0001
        else:
                lr = 0.00001
        return lr


with tf.device('/device:GPU:0'):
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
        classifier.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(48,48,3),activation='relu'))

        #Step 2 Pooling
        classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        #Adding 2nd Convolution Layer
        classifier.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(48,48,3),activation='relu'))
        classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))

        #adding 3rd Convoluation Layer
        classifier.add(tf.keras.layers.Conv2D(64,(3,3),input_shape=(48,48,3),activation='relu'))
        classifier.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))




 


        #Step 3 Flattening
        classifier.add(tf.keras.layers.Flatten())
        classifier.add(tf.keras.layers.Dropout(0.3))

        #Step 4 Full Connection
        classifier.add(tf.keras.layers.Dense(units=128,activation='relu'))
        classifier.add(tf.keras.layers.Dropout(0.3))
        classifier.add(tf.keras.layers.Dense(units=3,activation='softmax'))


        classifier.summary()

        #Compiling the CNN
        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
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

        training_set = train_datagen.flow_from_directory('dataset/new_training_set',
                                                 target_size = (48, 48),
                                                 batch_size = 128,
                                                 class_mode = 'categorical')

        test_set = test_datagen.flow_from_directory('dataset/new_test_set',
                                            target_size = (48, 48),
                                            batch_size = 128,
                                            shuffle=True,
                                            class_mode = 'categorical')
                                        


        classifier.fit_generator(training_set,
                         steps_per_epoch = 12000,
                         epochs = 15,
                         validation_data = test_set,
                         callbacks = callbacks_list,
                         validation_steps = 3000)
