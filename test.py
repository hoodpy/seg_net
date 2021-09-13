import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from model import Segnet


def vis_detection(image, label):
	#0=background 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow
	#11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person, 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=monitor
	label_colors = [np.array([[0, 0, 0]]), np.array([[128, 0, 0]]), np.array([[0, 128, 0]]), np.array([[128, 128, 0]]), 
	np.array([[0, 0, 128]]), np.array([[128, 0, 128]]), np.array([[0, 128, 128]]), np.array([[128, 128, 128]]), np.array([[64, 0, 0]]), 
	np.array([[192, 0, 0]]), np.array([[64, 128, 0]]), np.array([[192, 128, 0]]), np.array([[64, 0, 128]]), np.array([[192, 0, 128]]), 
	np.array([[64, 128, 128]]), np.array([[192, 128, 128]]), np.array([[0, 64, 0]]), np.array([[128, 64, 0]]), np.array([[0, 192, 0]]), 
	np.array([[128, 192, 0]]), np.array([[0, 64, 128]])]
	label = cv2.resize(label, (np.shape(image)[1], np.shape(image)[0]), interpolation=cv2.INTER_NEAREST)
	for i in range(1, 21):
		h_list, w_list = np.where(label==i)
		image[h_list, w_list, :] = 0.5 * image[h_list, w_list, :].astype(np.int32).astype(np.float32)+\
		0.5 * label_colors[i][:, (2, 1, 0)].astype(np.float32)
	return image.astype(np.uint8)


file_path = "D:/program/seg_net/demo"
image_shape = [224, 224]
network = Segnet(num_classes=21)
image_input = tf.placeholder(tf.uint8, [None, None, 3])
image_prepare = tf.image.convert_image_dtype(image_input, dtype=tf.float32)
image_prepare = tf.image.resize_images(image_prepare, image_shape, method=0)
image_prepare = tf.expand_dims(image_prepare, axis=0)
network.build_network(image_prepare, is_training=False)
saver = tf.compat.v1.train.Saver(var_list=tf.compat.v1.global_variables())
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement = True
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.run(tf.compat.v1.global_variables_initializer())
saver.restore(sess, "D:/program/seg_net/model/model58260.ckpt")
for name in os.listdir(file_path):
	image = cv2.imread(os.path.join(file_path, name))[:, :, ::-1]
	_, _, label = network.test_images(sess, image_input, image)
	fig, ax = plt.subplots(1, 1, figsize=(6, 6))
	ax.imshow(vis_detection(image, label[0]))
plt.show()

#cameraCapture = cv2.VideoCapture(0)
#cv2.namedWindow("001")
#res, frame = cameraCapture.read()
#while res and cv2.waitKey(1) != 27:
#	frame = cv2.flip(frame, 1)
#	_, _, label = network.test_images(sess, image_input, frame[:, :, (2, 1, 0)])
#	cv2.imshow("001", vis_detection(frame, label[0]))
#	res, frame = cameraCapture.read()
#cv2.destroyWindow("001")
#cameraCapture.release()