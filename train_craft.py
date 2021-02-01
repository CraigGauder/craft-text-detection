from SynthTextProcessor import *
from CRAFTNet import *
from CRAFTLoss import *

import os, sys
import pickle
import tensorflow as tf
# from tensorflow.keras.optimizers import adam
from sklearn.model_selection import KFold

def get_resized_input_and_output(x_data, y_data, size):
	def resize_individual_map(x, output_size):
		return tf.image.resize(x, size=[output_size, output_size])

	x_data_orig_sizes = [x.shape[:2] for x in x_data]
	y_data_orig_sizes = [y.shape[:2] for y in y_data]

	x_data = tf.Variable(tf.stack([tf.convert_to_tensor(resize_individual_map(x, size)) for x in x_data]))
	y_data = tf.stack([tf.convert_to_tensor(resize_individual_map(y, size//2)) for y in y_data])

	return x_data, x_data_orig_sizes, y_data, y_data_orig_sizes


if __name__ == '__main__':
	DATA_DIR = '/Users/CraigGauder/ML/craft-text-detection/data/SynthText'
	CHECKPOINT_PATH = "/Users/CraigGauder/ML/craft-text-detection/model/checkpoints/cp-{fold:04d}-{epoch:04d}.ckpt"
	EPOCHS = 10
	BATCH_SIZE = 100

	model = CRAFTNet()
	craft_loss = CRAFTLoss()

	model.save_weights(CHECKPOINT_PATH.format(epoch=0, fold=0))

	with open(os.path.join(DATA_DIR, 'dataImages.pickle'), 'rb') as file:
		x_data = pickle.load(file)
	with open(os.path.join(DATA_DIR, 'dataLabels.pickle'), 'rb') as file:
		y_data = pickle.load(file)

	x_data, x_shapes, y_data, y_shape = get_resized_input_and_output(x_data, y_data, 200)

	model.compile(
		loss=craft_loss,
		optimizer='adam',
		metrics=['accuracy']
	)


	cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHECKPOINT_PATH,
                                                 save_weights_only=True,
                                                 verbose=1)
	train_history = []
	test_history = []
	kf = KFold(n_splits=10, shuffle=True)
	for fold, (train_index, test_index) in enumerate(kf.split(x_data, y_data)):		
		print(f'training fold {fold+1}...')

		train_x = tf.gather(x_data, train_index)
		train_y = tf.gather(y_data, train_index)

		test_x = tf.gather(x_data, test_index)
		test_y = tf.gather(y_data, test_index)

		# Fit data to model
		train_history.append(model.fit(
			train_x, 
			train_y,
			batch_size=BATCH_SIZE,
			epochs=EPOCHS,
			validation_split=0.1,
			verbose=True,
			callbacks=[cp_callback]
		))

		test_history.append(model.evaluate(
			test_x,
			test_y
		))

		print(test_history[-1])

	model.save_weights("/Users/CraigGauder/ML/craft-text-detection/model/checkpoints/model_final.mdl")


