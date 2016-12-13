'''
Mentor - Mr.Prashant Kaushik 
Author - Hrishikesh Singh & Shivan Trivedi
'''

# Package import 
from FaceDetector import FaceDetector
from FaceModel import FaceModel
from Util import Util
from Display import Display
from Capture import Capture

# Import Libraries
import time, math, cProfile, numpy, cv2, subprocess
import cv2 as cv
from collections import deque
from PIL import Image , ImageOps , ImageEnhance
from scipy.cluster import vq
import matplotlib
import matplotlib.pyplot as plt

# Constants
CAMERA_INDEX = 0
SCALE_FACTOR = 5  # video size will be 1/SCALE_FACTOR
FACE_CLASSIFIER_PATH = "classifiers/haar-face.xml"
EYE_CLASSIFIER_PATH = "classifiers/haar-eyes.xml"
FACE_MIN_SIZE = 0.2
EYE_MIN_SIZE = 0.03

DISPLAY_SCALE = 0.3333
FACE_SCALE = 0.25
EYE_SCALE = 0.33333


def main():
    # Instantiate Classes
    detector = FaceDetector(FACE_CLASSIFIER_PATH, EYE_CLASSIFIER_PATH)
    model = FaceModel()
    display = Display()
    capture = Capture()

    oldTime = time.time()
    i = 0
    subprocess.call(['speech-dispatcher'])
    
    while True:
        # Calculate time difference (dt), update oldTime variable
        newTime = time.time()
        dt = newTime - oldTime
        oldTime = newTime

        # Grab Frames
        frames = capture.read()
        
        # Detect face 20% of the time, eyes 100% of the time
        if i % 10 is 0:
            rects = detector.detect(frames)
        else:
            rects = detector.detect(frames, model.getPreviousFaceRects())
        i += 1

        # Add detected rectangles to model
        model.add(rects)

        # Render
        display.renderScene(frames['display'], model, rects)
        display.renderEyes(frames['color'], model)
        # cv2.imshow("Video", frames['display'])
        
cProfile.run('main()', 'profile.o', 'cumtime')