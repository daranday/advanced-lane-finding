import pickle
from os.path import join, splitext, basename, exists, abspath

import cv2
import numpy as np

from utils import (CALIBRATION_DIR, OUTPUT_DIR, create_dirs, OpencvClicker)
from calibrate import CameraCalibrator

_warping_debug_dir = join(OUTPUT_DIR, 'warping')


class ImageWarper(object):
    """docstring for ImageWarper"""

    def __init__(self):
        self.reset()

    def reset(self, M=None):
        self.M = M

    def find_perspective_transform(self, src, dst):
        src = np.array(src, dtype=np.float32)
        dst = np.array(dst, dtype=np.float32)
        M = cv2.getPerspectiveTransform(src, dst)
        self.reset(M)

    def warp(self, img):
        assert self.M is not None
        return cv2.warpPerspective(img, self.M, img.shape[:2][::-1],
                                   flags=cv2.INTER_LINEAR)


class PerspectiveTransformFinder(object):
    """docstring for PerspectiveTransformFinder"""

    def __init__(self, debug_dir=_warping_debug_dir):
        self.title = 'Click to select 4 source points for warping.'
        self.title += ' Press r to reset'
        self.debug_dir = debug_dir
        self.perspective_transform_file = \
            join(debug_dir, 'perspective_transform.p')
        self.warper = ImageWarper()
        self.calibrator = CameraCalibrator()

    def initialize_image(self, image):
        cv2.namedWindow(self.title)
        self.clicker = OpencvClicker(self.title)
        self.calibrator.calibrate(CALIBRATION_DIR)
        self.image = self.calibrator.undistort(image)
        self.clone = None
        self.preview = None

        h, w, c = image.shape
        margin = w / 4.0
        self.keypoints = []
        self.dest_keypoints = [(margin, 0), (w - margin, 0),
                               (margin, h), (w - margin, h)]
        self.M = None

    def load(self):
        self.M = \
            pickle.load(open(self.perspective_transform_file, 'rb'))
        self.warper.reset(M=self.M)
        print('[i] Loading perspective transform from',
              self.perspective_transform_file)

    def save(self):
        create_dirs([self.debug_dir])
        self.warper.reset(M=self.M)
        pickle.dump(self.M,
                    open(self.perspective_transform_file, 'wb'))
        print('[i] Succesfully saved perspective tranform params to',
              self.perspective_transform_file)

    def find(self, image):
        try:
            self.load()
            return self.warper
        except:
            pass

        self.initialize_image(image)
        self.run()
        self.save()

        return self.warper

    def warp_image(self):
        self.warper.find_perspective_transform(
            self.keypoints,
            self.dest_keypoints)
        self.preview = self.warper.warp(self.image)
        self.M = self.warper.M

    def update_images(self):
        self.clone = self.image.copy()
        for p in self.keypoints:
            cv2.circle(self.clone, p, 1, (0, 0, 255), thickness=2)

        if len(self.keypoints) == 4:
            self.warp_image()
        else:
            self.preview = np.zeros_like(self.image)

        self.img = np.hstack([self.clone, self.preview])

    def show(self):
        self.update_images()
        cv2.imshow(self.title, self.img)
        key = chr(cv2.waitKey(100) & 0xFF)
        return key

    def run(self):
        key = None
        while True:
            key = self.show()

            if self.clicker.has_update():
                print('User clicked')
                self.keypoints.append(self.clicker.retrieve())

            if key == 'q':
                raise Exception('User quit perspective transform finder.')
            elif key == 's':
                if len(self.keypoints) == 4:
                    return
            elif key == 'r':
                self.keypoints = []
            elif key == 'b':
                self.keypoints.pop()
