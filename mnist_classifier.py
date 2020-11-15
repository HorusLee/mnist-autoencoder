# -*- coding: utf-8 -*-

"""
@author: Horus
Class:
Date: Thu Nov 12, 2020
Assignment 3
Description of Problem:
Mnist Classifier
"""

# import necessary libraries
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def get_train_test():
	"""get the normalized mnist dataset"""
	# load the mnist dataset
	(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	# choose 4000 data points for training set, 1000 data points for test set
	n_train, n_test = 4000, 1000
	x_tra = np.zeros((n_train, x_train.shape[1], x_train.shape[2]))
	y_tra = np.zeros(n_train)
	x_tes = np.zeros((n_test, x_test.shape[1], x_test.shape[2]))
	y_tes = np.zeros(n_test)

	# make sure each digit has equal number of points in each set
	train_num, test_num = np.zeros(10), np.zeros(10)
	m, n = 0, 0
	for i in range(len(x_train)):
		if train_num[y_train[i]] < n_train / 10:
			x_tra[m] = x_train[i]
			y_tra[m] = y_train[i]
			train_num[y_train[i]] += 1
			m += 1
		if m == n_train:
			break
	for i in range(len(x_test)):
		if test_num[y_test[i]] < n_test / 10:
			x_tes[n] = x_test[i]
			y_tes[n] = y_test[i]
			test_num[y_test[i]] += 1
			n += 1
		if n == n_test:
			break

	# normalize the data to [0, 1]
	x_tra, x_tes = x_tra / 255.0, x_tes / 255.0
	return x_tra, y_tra, x_tes, y_tes


# initialize some parameters
n_hidden_layers = 1
n_hidden_neurons = [128]
EPOCHS = 100

if __name__ == '__main__':
	# get the training and test dataset
	x_tr, y_tr, x_te, y_te = get_train_test()

	# train a 1-hidden layer neural network to recognize the digits
	model = tf.keras.models.Sequential([
		tf.keras.layers.Flatten(input_shape=x_tr.shape[1:])
	])
	for h in range(n_hidden_layers):
		model.add(tf.keras.layers.Dense(n_hidden_neurons[h], activation='relu'))
	model.add(tf.keras.layers.Dense(10))

	# compile and train the model
	optimizer = tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.5)
	loss_func = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
	model.compile(optimizer=optimizer, loss=loss_func, metrics=['accuracy'])
	model.summary()
	hist = model.fit(x_tr, y_tr, validation_data=(x_te, y_te), epochs=EPOCHS)

	# get one confusion matrix for the training set and another for the test set
	train_predictions = np.argmax(model.predict(x_tr), axis=1)
	train_confusion_matrix = tf.math.confusion_matrix(y_tr, train_predictions)
	test_predictions = np.argmax(model.predict(x_te), axis=1)
	test_confusion_matrix = tf.math.confusion_matrix(y_te, test_predictions)

	# get the loss and accuracy from history
	history = hist.history
	loss = history['loss']
	acc = history['accuracy']
	val_loss = history['val_loss']
	val_acc = history['val_accuracy']
	epochs = range(EPOCHS)

	# plot the loss
	plt.figure(figsize=(6, 9))
	plt.subplot(2, 1, 1)
	plt.plot(epochs, loss, label='Training Loss')
	plt.plot(epochs, val_loss, label='Validation loss')
	plt.grid(True)
	plt.legend()
	plt.title('Training and Validation Loss')
	plt.xlabel('Epochs')
	plt.ylabel('Loss')

	# plot the accuracy
	plt.subplot(2, 1, 2)
	plt.plot(epochs, acc, label='Training Accuracy')
	plt.plot(epochs, val_acc, label='Validation Accuracy')
	plt.grid(True)
	plt.legend()
	plt.title('Training and Validation Accuracy')
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	# plt.savefig('loss_and_accuracy.png')
	plt.show()
