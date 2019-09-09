import numpy as np
import keras
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import time
from sklearn.metrics import confusion_matrix
from keras.preprocessing.image import image
from pandas_ml import ConfusionMatrix
from tensorflow.keras.callbacks import TensorBoard, LearningRateScheduler
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Activation
from keras.layers.core import Dense, Flatten, Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import *
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from keras.models import load_model

def reconstruct_architecture(name):
	with open(name, 'r') as file:
		model = model_from_json(file.read())
		return model

#GPU configuration:
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.per_process_gpu_memory_fraction = 0.80
K.tensorflow_backend.set_session(tf.Session(config=configuration))

checkpoint_path = "models/gbm-classification-vgg16-lr-0.0001-dr-0.5-stamp-1555150592.ckpt"
fp_test = "positive-negative/test"

#RECONSTRUCTING THE NETWORK FOR PREDICTIONS
#Network config variables
width, height, channels = (224,224,3)
test_size = 592

#loading the test batch
test_batches = ImageDataGenerator(rescale=1./255).flow_from_directory(fp_test, 
    target_size=(width,height), 
    classes=["negative","positive"], 
    batch_size=1, 
    interpolation="bicubic", 
    shuffle=False)

#Importing and retraining VGG16 model 
VGG16 = keras.applications.vgg16.VGG16()
VGG16.summary()

#copying layers where dropouts will be added
fully_con_1 = VGG16.layers[-3]
fully_con_2 = VGG16.layers[-2]

#adding dropouts inbetween fully-connected layers
model = Dropout(.0001)(fully_con_1.output)
model = fully_con_2(model)
model = Dropout(.0001)(model)

#stitiching back the top of the model with the bottom of pre-trained network
classifier = Dense(2,activation="softmax", name="classifier")(model)
vgg16_binary = Model(input=VGG16.input, output=classifier, name="custom_vgg_classifier")
vgg16_binary.load_weights(checkpoint_path)

vgg16_binary.compile(Adam(lr=0.0001),
    loss="categorical_crossentropy",
    metrics = ["accuracy"])

vgg16_binary.summary()

predictions = vgg16_binary.predict_generator(test_batches, steps=test_size, verbose=1)
y_pred = np.argmax(predictions, axis=1)
test_loss, test_acc = vgg16_binary.evaluate_generator(test_batches, steps=test_size, verbose=1)

matrix = ConfusionMatrix(test_batches.classes, y_pred)
label_map = (test_batches.class_indices)
print(label_map)
print("Test Set Performance: Accuracy %f Loss %f" % (test_acc, test_loss))
matrix.plot()
plt.show()

