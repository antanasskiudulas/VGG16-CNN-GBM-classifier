import numpy as np
import keras
import itertools
import matplotlib.pyplot as plt
import tensorflow as tf
import time
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

#GPU configuration:
configuration = tf.ConfigProto()
configuration.gpu_options.allow_growth = True
configuration.gpu_options.per_process_gpu_memory_fraction = 0.8
K.tensorflow_backend.set_session(tf.Session(config=configuration))

#===============KERAS DEEP LEARNING MODEL CODE==============
#ALL THE DATA PATHS
fp_train = "positive-negative/train"
fp_test = "positive-negative/test"
fp_valid = "positive-negative/validation"
fp_models = "models"
#Network config variables
width, height, channels = (224,224,3)
train_size = 1184 
test_size = 592
val_size = 196
epochs = 400

#PARAMETERS TO CHECK ON TENSORBOARD FOR MODEL SELECTION
learning_rates = [.0001]
dropout_rates = [0.5]

for rate in learning_rates:
	for dropout in dropout_rates:
		model_name = 'gbm-classification-vgg16-lr-{}-dr-{}-stamp-{}'.format(rate, dropout, int(time.time()))
		#tensorboard visualisation
		tf_board = keras.callbacks.TensorBoard(log_dir='dump/{}'.format(model_name))
		cp_cb = tf.keras.callbacks.ModelCheckpoint("models/{}.ckpt".format(model_name),
	                                                 save_weights_only=True,
	                                                 monitor='val_loss',
	                                                 save_best_only=True,
	                                                 verbose=1)

		#Creating required batches (splitting batch sizes so that the batches run twice per epoch)
		train_batches = ImageDataGenerator(
			rescale=1./255,
		    rotation_range=10,
		    width_shift_range=0.1,
		    height_shift_range=0.1,
		    horizontal_flip=True,
		    fill_mode="nearest"
			).flow_from_directory(fp_train, target_size=(width,height), classes=["negative","positive"], batch_size=2, interpolation="bicubic")
		valid_batches = ImageDataGenerator(
			rescale=1./255
			).flow_from_directory(fp_valid, target_size=(width,height), classes=["negative","positive"], batch_size=2, interpolation="bicubic")

		#Importing and retraining VGG16 model 
		VGG16 = keras.applications.vgg16.VGG16()
		VGG16.summary()

		#copying layers where dropouts will be added
		fully_con_1 = VGG16.layers[-3]
		fully_con_2 = VGG16.layers[-2]

		#adding dropouts inbetween fully-connected layers
		model = Dropout(dropout)(fully_con_1.output)
		model = fully_con_2(model)
		model = Dropout(dropout)(model)

		#stitiching back the top of the model with the bottom of pre-trained network
		classifier = Dense(2,activation="softmax", name="classifier")(model)
		vgg16_binary = Model(input=VGG16.input, output=classifier, name="custom_vgg_classifier")

		for layer in vgg16_binary.layers[:-4]:
			layer.trainable = False

		vgg16_binary.summary()

		vgg16_binary.compile(Adam(lr=rate),
		  	loss="categorical_crossentropy",
		  	metrics = ["accuracy"])
		vgg16_binary.fit_generator(train_batches, 
		  	steps_per_epoch=1184, 
		  	validation_data=valid_batches, 
		  	validation_steps=196, 
		  	epochs=epochs,
		  	verbose=1,
		  	callbacks=[tf_board, cp_cb])
