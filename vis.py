import cv2
import numpy as np

cameraCapture = cv2.VideoCapture(0)
cv2.namedWindow("001")
res, frame = cameraCapture.read()
print(np.shape(frame))
while res and cv2.waitKey(1) != 27:
	cv2.imshow("001", frame)
	res, frame = cameraCapture.read()
cv2.destroyWindow("001")
cameraCapture.release()