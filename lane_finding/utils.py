import os
from os.path import abspath, join, realpath, dirname, exists
from glob import glob

import cv2

PROJECT_DIR = abspath(join(dirname(realpath(__file__)), '..'))
CALIBRATION_DIR = join(PROJECT_DIR, 'camera_cal')
OUTPUT_DIR = join(PROJECT_DIR, 'output')


def image_num(path):
    return int(path.partition('calibration')[2].partition('.')[0])


def get_chest_board_images():
    img_paths = sorted(glob(join(PROJECT_DIR, 'camera_cal/*')), key=image_num)
    return img_paths


def create_dirs(dirs):
    for d in dirs:
        if not exists(d):
            os.makedirs(d)


class OpencvClicker(object):
    """Class for enabling drag selection on a cv2 window."""

    def __init__(self, window_name):
        self.init_selecting()
        cv2.setMouseCallback(window_name, self.click)

    def init_selecting(self):
        """Resets drag state."""
        self.refPt = None

    def click(self, event, x, y, flags, param):
        """Mouse event handler.

        If the left mouse button was clicked, record the (x, y) coordinates.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            self.refPt = (x, y)
            print('[i] Point is:', self.refPt)

    def has_update(self):
        return self.refPt is not None

    def retrieve(self):
        tmp = self.refPt
        self.refPt = None
        return tmp
