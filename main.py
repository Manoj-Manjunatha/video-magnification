import cv2
import numpy as np


def build_gaussian_pyramid(frame, levels=3):
	# Build and return n-level gaussian pyramid.
	img = frame.copy()
	pyramid = []
	for i in range(levels):
		gaussian = cv2.pyrDown(img)
		pyramid.append(gaussian)
	return pyramid


def build_laplacian_pyramid(frame, levels=3):
	# Build and return n-level laplacian pyramid.
	gaussian_pyramid = build_gaussian_pyramid(frame, levels)
	pyramid = []

	# Traverse the gaussian pyramid image list in reverse order,
	# to build laplacian pyramid. 
	for i in range(levels - 1, 0, -1):
		gaussian_expanded = cv2.pyrUp(gaussian_pyramid[i])
		laplacian = cv2.subtract(gaussian_pyramid[i - 1], gaussian_expanded)
		pyramid.append(laplacian)
	return pyramid


def load_video(filename):
	capture = cv2.VideoCapture(filename)

	frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
	width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
	height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
	fps = int(capture.get(cv2.CAP_PROP_FPS))

	video_tensors = np.zeros((frames, height, width, 3), dtype='float')
	x = 0
	while capture.isOpened():
		ret_val, frame = capture.read()
		if ret_val:
			video_tensors[x] = frame
			laplacian_imgs = build_laplacian_pyramid(frame)
			for g_img in laplacian_imgs:
				cv2.imshow('laplacian_img', g_img)
				cv2.waitKey(100)	
			x += 1
			cv2.imshow('frame', frame)
			cv2.waitKey(1)
		else:
			break

	return video_tensors, fps



if __name__ == '__main__':
	v,fps = load_video('guitar.mp4')
	print(fps)
