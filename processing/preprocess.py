import cv2


def to_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)