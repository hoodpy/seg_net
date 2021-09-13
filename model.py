import tensorflow as tf
import time


class Timer():
	def __init__(self):
		self.total_time = 0
		self.calls = 0
		self.start_time = 0
		self.diff = 0
		self.average_time = 0

	def tic(self):
		self.start_time = time.time()

	def toc(self, average=True):
		self.diff = time.time() - self.start_time
		self.total_time += self.diff
		self.calls += 1
		self.average_time = self.total_time / self.calls
		if average:
			return self.average_time
		else:
			return self.diff


class Segnet():
	def __init__(self, num_classes):
		self._num_classes = num_classes

	def conv2d(self, images, out_channals, scope, is_training, kernel_size=3, stride=1, padding="SAME", bn=True, relu=True):
		with tf.compat.v1.variable_scope(scope):
			input_channals = images.get_shape().as_list()[3]
			weights = tf.compat.v1.get_variable("weights", [kernel_size, kernel_size, input_channals, out_channals], 
				initializer=tf.truncated_normal_initializer(stddev=0.1), trainable=is_training)
			biases = tf.compat.v1.get_variable("biases", [out_channals], initializer=tf.constant_initializer(0.0), trainable=is_training)
			conv_img = tf.nn.conv2d(images, weights, strides=[1, stride, stride, 1], padding=padding, name="conv2d")
			conv_img = tf.nn.bias_add(conv_img, biases, name="bias_add")
			if bn:
				conv_img = tf.layers.batch_normalization(conv_img, training=is_training, name="bn_op")
			if relu:
				conv_img = tf.nn.relu(conv_img, name="relu_op")
		return conv_img

	def up_sample(self, indics, updates, template, scope):
		with tf.compat.v1.variable_scope(scope):
			updates = tf.reshape(updates, [-1])
			output_shape = template.get_shape().as_list()
			batch_range = tf.reshape(tf.range(output_shape[0], dtype=indics.dtype), [output_shape[0], 1, 1, 1])
			b = tf.ones_like(indics) * batch_range
			b = tf.reshape(b, [-1, 1])
			indics_ = tf.reshape(indics, [-1, 1])
			indics_ = tf.concat([b, indics_], 1)
			ret = tf.scatter_nd(indics_, updates, shape=[output_shape[0], output_shape[1]*output_shape[2]*output_shape[3]])
			ret = tf.reshape(ret, output_shape)
		return ret

	def build_network(self, images, is_training):
		with tf.compat.v1.variable_scope("inference"):
			conv0_0 = self.conv2d(images, out_channals=64, scope="conv0_0", is_training=is_training)
			conv0_1 = self.conv2d(conv0_0, out_channals=64, scope="conv0_1", is_training=is_training)
			pool0_2, indics0_2 = tf.nn.max_pool_with_argmax(conv0_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="pool0_2")

			conv1_0 = self.conv2d(pool0_2, out_channals=128, scope="conv1_0", is_training=is_training)
			conv1_1 = self.conv2d(conv1_0, out_channals=128, scope="conv1_1", is_training=is_training)
			pool1_2, indics1_2 = tf.nn.max_pool_with_argmax(conv1_1, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="pool1_2")

			conv2_0 = self.conv2d(pool1_2, out_channals=256, scope="conv2_0", is_training=is_training)
			conv2_1 = self.conv2d(conv2_0, out_channals=256, scope="conv2_1", is_training=is_training)
			conv2_2 = self.conv2d(conv2_1, out_channals=256, scope="conv2_2", is_training=is_training)
			pool2_3, indics2_3 = tf.nn.max_pool_with_argmax(conv2_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="pool2_3")

			conv3_0 = self.conv2d(pool2_3, out_channals=512, scope="conv3_0", is_training=is_training)
			conv3_1 = self.conv2d(conv3_0, out_channals=512, scope="conv3_1", is_training=is_training)
			conv3_2 = self.conv2d(conv3_1, out_channals=512, scope="conv3_2", is_training=is_training)
			pool3_3, indics3_3 = tf.nn.max_pool_with_argmax(conv3_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="pool3_3")

			conv4_0 = self.conv2d(pool3_3, out_channals=512, scope="conv4_0", is_training=is_training)
			conv4_1 = self.conv2d(conv4_0, out_channals=512, scope="conv4_1", is_training=is_training)
			conv4_2 = self.conv2d(conv4_1, out_channals=512, scope="conv4_2", is_training=is_training)
			pool4_3, indics4_3 = tf.nn.max_pool_with_argmax(conv4_2, ksize=[1,2,2,1], strides=[1,2,2,1], padding="VALID", name="pool4_3")

			conv5_0 = self.up_sample(indics=indics4_3, updates=pool4_3, template=conv4_2, scope="upsample5_0")
			conv5_1 = self.conv2d(conv5_0, out_channals=512, scope="conv5_1", is_training=is_training)
			conv5_2 = self.conv2d(conv5_1, out_channals=512, scope="conv5_2", is_training=is_training)
			conv5_3 = self.conv2d(conv5_2, out_channals=512, scope="conv5_3", is_training=is_training)

			conv6_0 = self.up_sample(indics=indics3_3, updates=conv5_3, template=conv3_2, scope="upsample6_0")
			conv6_1 = self.conv2d(conv6_0, out_channals=512, scope="conv6_1", is_training=is_training)
			conv6_2 = self.conv2d(conv6_1, out_channals=512, scope="conv6_2", is_training=is_training)
			conv6_3 = self.conv2d(conv6_2, out_channals=256, scope="conv6_3", is_training=is_training)

			conv7_0 = self.up_sample(indics=indics2_3, updates=conv6_3, template=conv2_2, scope="upsample7_0")
			conv7_1 = self.conv2d(conv7_0, out_channals=256, scope="conv7_1", is_training=is_training)
			conv7_2 = self.conv2d(conv7_1, out_channals=256, scope="conv7_2", is_training=is_training)
			conv7_3 = self.conv2d(conv7_2, out_channals=128, scope="conv7_3", is_training=is_training)

			conv8_0 = self.up_sample(indics=indics1_2, updates=conv7_3, template=conv1_1, scope="upsample8_0")
			conv8_1 = self.conv2d(conv8_0, out_channals=128, scope="conv8_1", is_training=is_training)
			conv8_2 = self.conv2d(conv8_1, out_channals=64, scope="conv8_2", is_training=is_training)

			conv9_0 = self.up_sample(indics=indics0_2, updates=conv8_2, template=conv0_1, scope="upsample9_0")
			conv9_1 = self.conv2d(conv9_0, out_channals=64, scope="conv9_1", is_training=is_training)
			conv9_2 = self.conv2d(conv9_1, out_channals=64, scope="conv9_2", is_training=is_training)

			result = self.conv2d(conv9_2, out_channals=self._num_classes, scope="result", is_training=is_training, 
				kernel_size=1, bn=False, relu=False)
			result_softmax = tf.nn.softmax(result, axis=-1, name="result_softmax")
			result_argmax = tf.argmax(result, dimension=-1, name="result_argmax")

			self._result = result
			self._result_softmax = result_softmax
			self._result_argmax = result_argmax

	def add_loss(self, annotations):
		labels, annotations = tf.reshape(self._result, [-1, self._num_classes]), tf.reshape(annotations, [-1])
		effective_indics = tf.squeeze(tf.compat.v2.where(tf.math.less_equal(annotations, self._num_classes-1)), 1)
		prediction, ground_truth = tf.gather(labels, effective_indics), tf.gather(annotations, effective_indics)
		cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prediction, labels=ground_truth), 
			name="cross_entropy")
		self._cross_entropy = cross_entropy
		return cross_entropy

	def train_step(self, sess, train_op, global_step, merged):
		_, _cross_entropy, step, summary = sess.run([train_op, self._cross_entropy, global_step, merged])
		return _cross_entropy, step, summary

	def test_images(self, sess, input_op, images):
		feed_dict = {input_op: images}
		result, result_softmax, result_argmax = sess.run([self._result, self._result_softmax, self._result_argmax], feed_dict=feed_dict)
		return result, result_softmax, result_argmax
