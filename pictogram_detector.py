"""Small Script For Detecting Pictograms in Images"""
import cv2
import numpy as np
from resizer import constant_aspect_resize


def detect_pictogram(path):
    """Main Function that detects the Pictogram"""
    ker_list = [np.ones((99, 1), np.uint8), np.ones((1, 99), np.uint8)]

    image = cv2.imread(path)
    image = constant_aspect_resize(image, width=2500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    binary = 255 - gray
    vertical_check = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker_list[0])
    hor_check = cv2.morphologyEx(binary, cv2.MORPH_OPEN, ker_list[1])
    image[(vertical_check == 255) | (hor_check == 255), :] = 255
    return image
