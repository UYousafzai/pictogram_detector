"""UTILITIES:
    helper/utility functions required by main processing modules
"""
import cv2


def constant_aspect_resize(image, width=2500, height=None):
    """A simple image resizing function"""
    (_h, _w) = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height / float(_h)
        dim = (int(_w * r), height)
    else:
        r = width / float(_w)
        dim = (width, int(_h * r))
    if dim[0] < _w:
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    else:
        resized = cv2.resize(image, dim, interpolation=cv2.INTER_LANCZOS4)
    return resized
