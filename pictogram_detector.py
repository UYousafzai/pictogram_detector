"""Small Script For Detecting Pictograms in Images"""
import cv2
import numpy as np
import pytesseract
from utility import constant_aspect_resize


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


if __name__ == "__main__":
    for i in range(1, 10):
        rpath = "data/" + str(i) + ".png"
        wpath = "output/" + str(i) + ".png"
        img = detect_pictogram(rpath)
        d = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
        n_boxes = len(d["level"])
        for i in range(n_boxes):
            if (d["level"][i] > 4) and (d["height"][i] < img.shape[0] / 2):
                (x, y, w, h) = (
                    d["left"][i],
                    d["top"][i],
                    d["width"][i],
                    d["height"][i],
                )
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        cv2.imwrite(wpath, img)
