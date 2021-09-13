import numpy as np
import cv2
import os


masks_path = "E:/VOCdevkit/VOC2012/SegmentationClass"
save_path = "E:/VOCdevkit/VOC2012/SegmentationLabel"
labels_list = [(0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128), (128,0,128), (0,128,128), 
(128,128,128), (64,0,0), (192,0,0), (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128), 
(192,128,128), (0,64,0), (128,64,0), (0,192,0), (128,192,0), (0,64,128)]
files_name = [name for name in os.listdir(masks_path)]
pixels_num = [0] * len(labels_list)
for name in files_name:
	image = cv2.imread(os.path.join(masks_path, name))[:, :, (2, 1, 0)]
	high, width, depth = np.shape(image)
	label = np.zeros((high, width))
	for i in range(high):
		for j in range(width):
			the_val = tuple(image[i, j, :])
			if the_val in labels_list:
				index = labels_list.index(the_val)
				label[i, j] = index
				pixels_num[index] += 1
	cv2.imwrite(os.path.join(save_path, name), label)
print(pixels_num)
