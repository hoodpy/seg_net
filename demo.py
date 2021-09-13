import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from model import Segnet


def vis_detection(image, label):
	#0=background 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow
	#11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=monitor
	label_colors = [(0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128), (128,0,128), (0,128,128),
	(128,128,128), (64,0,0), (192,0,0), (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128),
	(192,128,128), (0,64,0), (128,64,0), (0,192,0), (128,192,0), (0,64,128)]
	annotation = np.zeros_like(image)
	for i in range(1, 21):
		h_list, w_list = np.where(label==i)
		annotation[h_list, w_list, :] = label_colors[i]
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 6))
	ax1.imshow(image)
	ax2.imshow(annotation)


image_size = [224, 224]
num_classes = 21
demo_path = "D:/program/seg_net/demo"
images_path = [os.path.join(demo_path, name) for name in os.listdir(demo_path)]
model_path = "D:/program/seg_net/model/model120310.ckpt"
network = Segnet(num_classes=num_classes)
image_input = tf.compat.v1.placeholder(tf.uint8, shape=[None, None, 3])
image_perpare = tf.image.convert_image_dtype(image_input, dtype=tf.float32)
image_perpare = tf.image.resize_images(image_perpare, image_size, method=0)
image_perpare = tf.expand_dims(image_perpare, axis=0)
network.build_network(image_perpare, is_training=False)
saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())


if __name__ == "__main__":
	config = tf.compat.v1.ConfigProto()
	config.allow_soft_placement = True
	config.gpu_options.allow_growth = True
	sess = tf.compat.v1.Session(config=config)
	sess.run(tf.compat.v1.global_variables_initializer())
	saver.restore(sess, model_path)
	for image_path in images_path:
		image = cv2.imread(image_path)[:, :, (2,1,0)]
		_, _, label = network.test_images(sess, image_input, image)
		image = cv2.resize(image, (image_size[0], image_size[1]))
		vis_detection(image, label[0])
	plt.show()