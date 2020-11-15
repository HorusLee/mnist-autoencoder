# -*- coding: utf-8 -*-

"""
@author: Horus
Class:
Date: Thu Nov 12, 2020
Assignment
Description of Problem:
Normal AE and De-noising AE
"""

# import necessary libraries
from abc import ABC

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

# import the get_train_test function from problem 1
from mnist_classifier import get_train_test


class Autoencoder(tf.keras.models.Model, ABC):
	"""define an autoencoder with two Dense layers: an encoder and a decoder"""
	def __init__(self, latent_dim):
		super(Autoencoder, self).__init__()
		self.latent_dim = latent_dim
		self.encoder = tf.keras.Sequential([
			tf.keras.layers.Flatten(),
			tf.keras.layers.Dense(latent_dim, activation='relu'),
		])
		self.decoder = tf.keras.Sequential([
			tf.keras.layers.Dense(784, activation='sigmoid'),
			tf.keras.layers.Reshape((28, 28))
		])

	def call(self, x):
		"""the call function for autoencoder"""
		encoded = self.encoder(x)
		decoded = self.decoder(encoded)
		return decoded


def train_plot_AE(x_tr, x_te, original=True):
	"""create, compile and train the AE, then plot the images"""
	# create, compile and train the autoencoder
	autoencoder = Autoencoder(latent_dimension)
	optimizer = tf.keras.optimizers.SGD(learning_rate=2.0, momentum=0.5)
	loss_func = tf.keras.losses.MeanSquaredError()
	autoencoder.compile(optimizer=optimizer, loss=loss_func)
	callback = tf.keras.callbacks.EarlyStopping(patience=1)
	hist = autoencoder.fit(x_tr, x_tr, epochs=EPOCHS,
	                       validation_data=(x_te, x_te), callbacks=[callback])

	# get the loss from history
	history = hist.history
	loss = history['loss']
	val_loss = history['val_loss']
	epochs = range(len(loss))

	# plot the loss
	plt.figure(figsize=(6, 6))
	plt.plot(epochs, loss, label='Training Loss')
	plt.plot(epochs, val_loss, label='Validation loss')
	plt.grid(True)
	plt.legend()
	plt.title('Training and Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')
	plt.savefig('ae_loss_o.png' if original else 'ae_loss_n.png')
	plt.show()

	# choose 8 samples randomly from the test set, plot the original (input)
	# image and the output image produced by the network after training
	encoded_images = autoencoder.encoder(x_te).numpy()
	decoded_images = autoencoder.decoder(encoded_images).numpy()

	plt.figure(figsize=(2 * n, 4))
	for i in range(n):
		# display original
		ax = plt.subplot(2, n, i + 1)
		plt.imshow(x_te[i])
		plt.title('original' if original else 'original + noise')
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)

		# display reconstruction
		ax = plt.subplot(2, n, i + 1 + n)
		plt.imshow(decoded_images[i])
		plt.title('reconstructed')
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.savefig('ae_image_o.png' if original else 'ae_image_n.png')
	plt.show()
	return autoencoder


def feature_plot(input_hidden_layer, original=True):
	"""plot the features of each neuron for the hidden layer"""
	features = input_hidden_layer.get_weights()[0].transpose(). \
		reshape((latent_dimension, 28, 28))
	# plot the normalized clipped features
	features = (features - features.min()).clip(0, 1)
	plt.figure(figsize=(16, 8))
	for i in range(len(features)):
		ax = plt.subplot(8, 16, i + 1)
		plt.imshow(features[i])
		plt.gray()
		ax.get_xaxis().set_visible(False)
		ax.get_yaxis().set_visible(False)
	plt.savefig('feature_o.png' if original else 'feature_n.png')
	plt.show()


def semi_trained_classifier(input_hidden_layer, original=True):
	"""
	train a network where the weights from input to hidden neurons are set to
	the final values of the same weights from the best final network of AE,
	and output the confusion matrices and plot the time series of the error
	"""
	input_hidden_layer.trainable = False
	semi_trained_model = tf.keras.Sequential([
		input_hidden_layer,
		tf.keras.layers.Dense(10)
	])
	optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.5)
	loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	semi_trained_model.compile(optimizer=optimizer,
	                           loss=loss_func, metrics=['accuracy'])
	hist = semi_trained_model.fit(
		x_train, y_train, validation_data=(x_test, y_test), epochs=100)

	# get one confusion matrix for the training set and another for the test set
	train_predictions = np.argmax(semi_trained_model.predict(x_train), axis=1)
	train_confusion_matrix = tf.math.confusion_matrix(y_train, train_predictions)
	test_predictions = np.argmax(semi_trained_model.predict(x_test), axis=1)
	test_confusion_matrix = tf.math.confusion_matrix(y_test, test_predictions)

	# plot the time series of the error (1 - accuracy)
	history = hist.history
	error = 1 - np.array(history['accuracy'])
	val_error = 1 - np.array(history['val_accuracy'])
	epochs = range(100)

	# plot the accuracy
	plt.figure(figsize=(6, 6))
	plt.plot(epochs, error, label='Training Error')
	plt.plot(epochs, val_error, label='Validation Error')
	plt.grid(True)
	plt.legend()
	plt.title('Training and Validation Error')
	plt.xlabel('Epochs')
	plt.ylabel('Error')
	plt.savefig('classifier_error_o.png' if original else 'classifier_error_n')
	plt.show()

	return train_confusion_matrix, test_confusion_matrix


# initialize some parameters
EPOCHS = 200
latent_dimension = 128
n = 8
noise_factor = 0.2

if __name__ == '__main__':
	# get the training and test dataset
	x_train, y_train, x_test, y_test = get_train_test()

	# train the AE with original training data
	autoencoder_o = train_plot_AE(x_train, x_test)

	# adding random noise to the images
	x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
	x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
	x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0.,
	                                 clip_value_max=1.)
	x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0.,
	                                clip_value_max=1.)

	# train the AE with original + noise training data
	autoencoder_n = train_plot_AE(x_train_noisy, x_test_noisy, False)

	# put the input-to-hidden weights from the autoencoder into the classifier
	input_hidden_layer_o = autoencoder_o.get_layer("sequential")
	input_hidden_layer_n = autoencoder_n.get_layer("sequential_2")

	# plot the features for the hidden neurons
	feature_plot(input_hidden_layer_o)
	feature_plot(input_hidden_layer_n, False)

	# output the confusion matrix for each dataset
	train_confusion_matrix_o, test_confusion_matrix_o = \
		semi_trained_classifier(input_hidden_layer_o)
	train_confusion_matrix_n, test_confusion_matrix_n = \
		semi_trained_classifier(input_hidden_layer_n, False)
	print(train_confusion_matrix_o)
	print(test_confusion_matrix_o)
	print(train_confusion_matrix_n)
	print(test_confusion_matrix_n)
