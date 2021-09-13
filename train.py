import tensorflow as tf
import os
import numpy as np
from model import Timer, Segnet
import matplotlib.pyplot as plt


class Trainer():
	def __init__(self):
		self._file_path = "D:/program/seg_net/data/data.tfrecords"
		self._model_path = "D:/program/seg_net/model/"
		self._log_path = "D:/program/seg_net/log/"
		self._batch_size = 5
		self._image_size = [224, 224]
		self._num_classes = 21
		self._shuffle_size = 10
		self._epochs = 100
		self._learning_rate = 1e-4
		self.network = Segnet(num_classes=self._num_classes)
		self.timer = Timer()

	def parser(self, record):
		features = tf.io.parse_single_example(record, features={
			"high": tf.io.FixedLenFeature([], tf.int64), 
			"width": tf.io.FixedLenFeature([], tf.int64),
			"depth": tf.io.FixedLenFeature([], tf.int64),
			"image": tf.io.FixedLenFeature([], tf.string),
			"label": tf.io.FixedLenFeature([], tf.string)
			})
		high, width = tf.cast(features["high"], tf.int32), tf.cast(features["width"], tf.int32)
		depth = tf.cast(features["depth"], tf.int32)
		decode_image, decode_label = tf.decode_raw(features["image"], tf.uint8), tf.decode_raw(features["label"], tf.uint8)
		decode_image, decode_label = tf.reshape(decode_image, [high, width, depth]), tf.reshape(decode_label, [high, width, 1])
		return decode_image, decode_label

	def preprocess_for_train(self, image, label):
		image = tf.image.convert_image_dtype(image, dtype=tf.float32)
		label = tf.cast(label, tf.int32)
		image_shape = tf.shape(image)
		scale = tf.random.uniform([1], minval=0.5, maxval=2.0)
		new_h = tf.cast(tf.multiply(tf.cast(image_shape[0], tf.float32), scale), tf.int32)
		new_w = tf.cast(tf.multiply(tf.cast(image_shape[1], tf.float32), scale), tf.int32)
		new_shape = tf.squeeze(tf.stack([new_h, new_w]), axis=1)
		image = tf.image.resize_images(image, new_shape, method=0)
		label = tf.image.resize_images(label, new_shape, method=1)
		label = tf.cast(label - 255, tf.float32)
		total = tf.concat([image, label], -1)
		total_pad = tf.image.pad_to_bounding_box(total, 0, 0, tf.math.maximum(new_shape[0], self._image_size[0]), 
			tf.math.maximum(new_shape[1], self._image_size[1]))
		total_crop = tf.image.random_crop(total_pad, [self._image_size[0], self._image_size[1], 4])
		image, label = total_crop[:, :, :3], total_crop[:, :, 3]
		label = tf.cast(label + 255., tf.int32)
		image.set_shape([self._image_size[0], self._image_size[1], 3])
		label.set_shape([self._image_size[0], self._image_size[1]])
		return image, label

	def get_dataset(self):
		dataset = tf.data.TFRecordDataset(self._file_path)
		dataset = dataset.map(self.parser)
		dataset = dataset.map(lambda image, label: self.preprocess_for_train(image, label))
		dataset = dataset.shuffle(self._shuffle_size).repeat(self._epochs).batch(self._batch_size)
		self.iterator = dataset.make_initializable_iterator()
		image_batch, label_batch = self.iterator.get_next()
		return image_batch, label_batch

	def train(self):
		config = tf.compat.v1.ConfigProto()
		config.allow_soft_placement = True
		config.gpu_options.allow_growth = True
		with tf.compat.v1.Session(config=config) as sess:
			global_step = tf.Variable(0, trainable=False)
			learning_rate = tf.Variable(self._learning_rate, trainable=False)
			tf.compat.v1.summary.scalar("learning_rate", learning_rate)

			image_batch, label_batch = self.get_dataset()
			image_batch = tf.reshape(image_batch, [self._batch_size, self._image_size[0], self._image_size[1], 3])
			label_batch = tf.reshape(label_batch, [self._batch_size, self._image_size[0], self._image_size[1]])
			self.network.build_network(image_batch, is_training=True)

			cross_entropy = self.network.add_loss(label_batch)
			tf.compat.v1.summary.scalar("cross_entropy", cross_entropy)

			update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				train_op = tf.compat.v1.train.AdamOptimizer(learning_rate).minimize(cross_entropy, global_step=global_step)

			self.saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables(), max_to_keep=5)
			merged = tf.compat.v1.summary.merge_all()
			summary_writer = tf.compat.v1.summary.FileWriter(self._log_path, sess.graph)

			sess.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.local_variables_initializer()])
			sess.run(self.iterator.initializer)
			sess.run([tf.compat.v1.assign(learning_rate, self._learning_rate), tf.compat.v1.assign(global_step, 0)])

			while True:
				try:
					self.timer.tic()
					_cross_entropy, step, summary = self.network.train_step(sess, train_op, global_step, merged)
					summary_writer.add_summary(summary, step)
					self.timer.toc()
					if (step + 1) % 46608 == 0:
						sess.run(tf.compat.v1.assign(learning_rate, self._learning_rate * 0.1))
					if (step + 1) % 971 == 0:
						print(">>>Step: %.d\n>>>Cross_entropy: %.6f\n>>>Average_time: %.6fs\n" % (step + 1, _cross_entropy, 
							self.timer.average_time))
					if (step + 1) % 11652 == 0:
						self.snap_shot(sess, step + 1)
				except tf.errors.OutOfRangeError:
					break

			summary_writer.close()

	def snap_shot(self, sess, step):
		network = self.network
		self.saver.save(sess, os.path.join(self._model_path, "model%d.ckpt" % (step)))
		print("Wrote snapshot to: " + self._model_path + "model%d.ckpt\n" % (step))


if __name__ == "__main__":
	trainer = Trainer()
	trainer.train()