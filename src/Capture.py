'''
Mentor - Mr.Prashant Kaushik 
Author - Hrishikesh Singh & Shivan Trivedi
'''

# Import Libraries
import time, math, cProfile, numpy, cv2 ,subprocess
import cv2 as cv
from collections import deque
from PIL import Image , ImageOps , ImageEnhance
from scipy.cluster import vq
import matplotlib
import matplotlib.pyplot as plt

# Constants
CAMERA_INDEX = 0
SCALE_FACTOR = 20  # video size will be 1/SCALE_FACTOR
FACE_CLASSIFIER_PATH = "classifiers/haar-face.xml"
EYE_CLASSIFIER_PATH = "classifiers/haar-eyes.xml"
FACE_MIN_SIZE = 0.2
EYE_MIN_SIZE = 0.03

DISPLAY_SCALE = 0.3333
FACE_SCALE = 0.25
EYE_SCALE = 0.33333

class Capture:
	camera = cv2.VideoCapture(CAMERA_INDEX)
	height = 0
	width = 0
	
	def __init__(self, scaleFactor = 100 ):
		# Setup webcam dimensions
		self.height = self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT)
		self.width = self.camera.get(cv2.CAP_PROP_FRAME_WIDTH)
		# self.camera = self.camera.set(cv2.CAP_PROP_FPS,10)
		# Reduce Video Size to make Processing Faster
		if scaleFactor is not 1:
			scaledHeight = self.height / scaleFactor
			scaledWidth = self.width / scaleFactor
			self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, scaledHeight)
			self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, scaledWidth)

		# Create window
		cv2.namedWindow("Video", cv2.WINDOW_AUTOSIZE)
		
	def read(self):
		retVal, colorFrame = self.camera.read()
		cv2.imshow('Live',colorFrame)
		displayFrame = cv2.resize(colorFrame, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
		grayFrame = cv2.equalizeHist(cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY))
		faceFrame = cv2.resize(grayFrame, None, fx=FACE_SCALE, fy=FACE_SCALE)
		eyesFrame = cv2.resize(cv2.equalizeHist(cv2.cvtColor(colorFrame, cv2.COLOR_BGR2GRAY)), None, fx=EYE_SCALE, fy=EYE_SCALE)

		frames = {
			'color': colorFrame,
			'display': displayFrame,
			'gray': grayFrame,
			'face': faceFrame,
			'eyes': eyesFrame
		}

		return frames

