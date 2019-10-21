# CNNModelTraining

This is an simple machine learning model to train a CNN model using the library from tensorflow.keras.

Models are saved as H5 models

# prediction

After training and if you want to test your model prediction, you could make use of the model_prediction.py(Currently no flags have been added hence required to go into the codes and change to the path to your model)

# Convert to TFLite

you could make use of the tfLifeConvertor.py to convert your model into TFLITE model for your mobile or Mircoprocessors

# Image Reizing

You can make use of resize.py to resize your image for optimal performance

# Dataset

Put your images in different folders as a indicator of labels to train for example, training dataset would be under dataset/train/"NAME OF THE LABEL" and for validation, dataset/test/"NAME OF THE LABEL"
