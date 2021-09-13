import tensorflow as tf
import numpy as np
import os
import cv2


def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

images_path = "E:/VOCdevkit/VOC2012/JPEGImages"
labels_path = "E:/VOCdevkit/VOC2012/SegmentationLabel"
files_name = [name.rstrip(name.split(".")[-1]) for name in os.listdir(labels_path)]
writer = tf.io.TFRecordWriter("D:/program/seg_net/data/data.tfrecords")

for name in files_name:
	image = cv2.resize(cv2.imread(os.path.join(images_path, name + "jpg"))[:, :, (2, 1, 0)], (224, 224))
	label = cv2.resize(cv2.imread(os.path.join(labels_path, name + "png"), 0), (224, 224), interpolation=cv2.INTER_NEAREST)
	shape = np.array(np.shape(image)).astype(np.int64)
	image, label = image.tostring(), label.tostring()
	example = tf.train.Example(features=tf.train.Features(feature={
		"high": _int64_feature(shape[0]), 
		"width": _int64_feature(shape[1]), 
		"depth": _int64_feature(shape[2]),
		"image": _bytes_feature(image),
		"label": _bytes_feature(label)
		}))
	writer.write(example.SerializeToString())

writer.close()