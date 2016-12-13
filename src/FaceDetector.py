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
SCALE_FACTOR = 20  # video size will be 1/SCALE_FACTOR
FACE_CLASSIFIER_PATH = "classifiers/haar-face.xml"
EYE_CLASSIFIER_PATH = "classifiers/haar-eyes.xml"
FACE_MIN_SIZE = 0.2
EYE_MIN_SIZE = 0.03

DISPLAY_SCALE = 0.3333
FACE_SCALE = 0.25
EYE_SCALE = 0.33333


class FaceDetector:
    """
	FaceDetector is a wrapper for the cascade classifiers.
	Must be initialized using faceClassifierPath and eyeClassifierPath, and 
	should only be initialized once per program instance. The only "public"
	method is detect().
	"""

    def __init__(self, faceClassifierPath, eyeClassifierPath):
        self.faceClassifier = cv2.CascadeClassifier(faceClassifierPath)
        self.eyeClassifier = cv2.CascadeClassifier(eyeClassifierPath)

    def detect(self, frames, faceRect=False):
        

        # Data structure to hold frame info
        rects = {
            'face': numpy.array([], dtype=numpy.int64),
            'eyeLeft': numpy.array([], dtype=numpy.int64),
            'eyeRight': numpy.array([], dtype=numpy.int64)
        }

        # Detect face if old faceRect not provided
        if faceRect is False or len(faceRect) is 0:
            faceIMG = frames['face']
            faceRects = self.classifyFace(faceIMG)

            # Ensure a single face found
            if len(faceRects) is 1:
                faceRect = faceRects[0]
            else:
                # TODO throw error message
                print('\x1b[1;37;41m' + 'H E A D    N O T    I N   C O R R E C T    P O S I T I O N' + '\x1b[0m')
                # print(" No Faces / Multiple Faces Found!")
                # subprocess.call(['speech-dispatcher'])        #start speech dispatcher
                # subprocess.call(['spd-say', '"alert"'])
                return rects

        rects['face'] = faceRect

        # Extract face coordinates, calculate center and diameter
        x1, y1, x2, y2 = rects['face']
        faceCenter = (((x1 + x2) / 2.0), ((y1 + y2) / 2.0))
        faceDiameter = y2 - y1

        # Extract eyes region of interest (ROI), cropping mouth and hair
        eyeBBox = numpy.array([x1, (y1 + (faceDiameter * 0.24)), x2, (y2 - (faceDiameter * 0.40))], dtype=numpy.int64)
        #		eyesY1 = (y1 + (faceDiameter * 0.16))
        #		eyesY2 = (y2 - (faceDiameter * 0.32))
        #		eyesX1 = x1 * EYE_SCALE
        #		eyesX2 = x2 * EYE_SCALE
        #		eyesROI = img[eyesY1:eyesY2, x1:x2]

        # Search for eyes in ROI
        eyeRects = self.classifyEyes(frames['eyes'], eyeBBox)
        #		print eyeRects

        # Ensure (at most) two eyes found
        if len(eyeRects) > 2:
            # TODO throw error message (and perhaps return?)
            print("Multiple Eyes Found!")
        # TODO get rid of extras by either:
        #	a) using two largest rects or
        #	b) finding two closest matches to average eyes

        # Loop over each eye
        for e in eyeRects:
            # Adjust coordinates to be in faceRect's coordinate space
            #			e += numpy.array([eyesX1, eyesY1, eyesX1, eyesY1],dtype=numpy.int64)

            # Split left and right eyes. Compare eye and face midpoints.
            eyeMidpointX = (e[0] + e[2]) / 2.0
            if eyeMidpointX < faceCenter[0]:
                rects['eyeLeft'] = e  # TODO prevent overwriting
            else:
                rects['eyeRight'] = e
        # TODO error checking
        # TODO calculate signal quality
        # print('final rects=', rects)
        left_eye = rects['eyeLeft']
        right_eye = rects['eyeRight']
        face_coor = rects['face']
        if len(right_eye) == 0 and len(left_eye) == 0:
            print('\x1b[1;37;41m' + 'Y O U   A R E   S L E E P I N G' + '\x1b[0m')
        elif len(left_eye) != 0 and len(right_eye) != 0:
            print('\x1b[6;30;42m' + '\t\t\tA W A K E\t\t' + '\x1b[0m')
        elif len(right_eye) == 0 or len(left_eye) == 0:
            print('\x1b[1;37;43m' + 'P a r t i a l    A t t e n t i o n' + '\x1b[0m')
                
    

        
        return rects

    def classify(self, img, cascade, minSizeX=40):
        """Run Cascade Classifier on Image"""
        minSizeX = int(round(minSizeX))
        #		print 'minSizeX:',minSizeX
        # Run Cascade Classifier
        rects = cascade.detectMultiScale(
            img, minSize=(minSizeX, minSizeX),
            flags=cv2.CASCADE_SCALE_IMAGE)

        # No Results
        if len(rects) == 0:
            return numpy.array([], dtype=numpy.int64)

        rects[:, 2:] += rects[:, :2]  # ? ? ?
        rects = numpy.array(rects, dtype=numpy.int64)
        return rects

    def classifyFace(self, img):
        """Run Face Cascade Classifier on Image"""
        rects = self.classify(img, self.faceClassifier, img.shape[1] * FACE_MIN_SIZE)
        return rects / FACE_SCALE

    def classifyEyes(self, img, bBox):
        """Run Eyes Cascade Classifier on Image"""
        EYE_MIN_SIZE = 0.15
        bBoxScaled = bBox * EYE_SCALE
        eyesROI = img[bBoxScaled[1]:bBoxScaled[3], bBoxScaled[0]:bBoxScaled[2]]

        eyesROI = cv2.equalizeHist(eyesROI)

        #		print 'eyesROI dimensions: ',eyesROI.shape
        minEyeSize = eyesROI.shape[1] * EYE_MIN_SIZE
        #		print 'minEyeSize:',minEyeSize
        # cv2.imshow("eyesROI", eyesROI)
        rectsScaled = self.classify(eyesROI, self.eyeClassifier,
                                    minEyeSize)

        #		print rectsScaled
        # Scale back to full size
        rects = rectsScaled / EYE_SCALE

        # Loop over each eye
        for eye in rects:
            # Adjust coordinates to be in faceRect's coordinate space
            eye += numpy.array([bBox[0], bBox[1], bBox[0], bBox[1]])

        return rects

