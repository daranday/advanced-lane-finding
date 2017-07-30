import sys
import json
from os.path import exists

import cv2
import numpy as np

import gflags

gflags.DEFINE_string('image_path',
                     '',
                     'Path to video')
gflags.DEFINE_string('hls_range_path',
                     '',
                     'Path to hls_range')
gflags.DEFINE_string('old_image_path',
                     '',
                     'Path to hls_range')
FLAGS = gflags.FLAGS


class ClickSelector(object):
    """Class for enabling drag selection on a cv2 window."""

    def __init__(self, window_name):
        self.refPt = None
        cv2.setMouseCallback(window_name, self.click)

    def clear(self):
        self.refPt = None

    def click(self, event, x, y, flags, param):
        """Mouse event handler.

        If the left mouse button was clicked, record the starting
        (x, y) coordinates and indicate that cropping is being
        performed.
        """
        if event == cv2.EVENT_LBUTTONUP:
            # Record the ending (x, y) coordinates and indicate that
            # the cropping operation is finished.
            self.refPt = (x, y)
            print('[i] Clicked:', self.refPt)


def paint_by_hls(hls_image, bgr_image, hls_min, hls_max):
    a = 0.8
    b = 1.
    c = 0.

    mask = cv2.inRange(hls_image, hls_min, hls_max)
    purple = np.zeros(bgr_image.shape, np.uint8)
    purple[:] = (238, 130, 238)
    purple = cv2.bitwise_and(purple, purple, mask=mask)
    painted_image = cv2.addWeighted(bgr_image, a, purple, b, c)

    return painted_image


def main():
    picker_window = 'Pick color'
    cv2.namedWindow(picker_window, cv2.WINDOW_NORMAL)
    image = cv2.imread(FLAGS.image_path)
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
    click_selector = ClickSelector(picker_window)
    clone = image.copy()
    old_image = None
    old_hls_image = None
    old_clone = None

    history = []
    hls_min = (180, 255, 255)
    hls_max = (0, 0, 0)
    hls_range_file = 'hls_range.json'

    if FLAGS.hls_range_path:
        assert FLAGS.old_image_path
        old_image = cv2.imread(FLAGS.old_image_path)
        old_hls_image = cv2.cvtColor(old_image, cv2.COLOR_BGR2HLS)
        hls_json = json.load(open(FLAGS.hls_range_path))
        hls_min = tuple(hls_json['hls_min'])
        hls_max = tuple(hls_json['hls_max'])
        clone = paint_by_hls(hls_image, image, hls_min, hls_max)
        old_clone = paint_by_hls(old_hls_image, old_image, hls_min, hls_max)

    while True:
        cv2.imshow(picker_window, clone)
        if old_clone is not None:
            cv2.imshow('Original image', old_clone)
        key = chr(cv2.waitKey(100) & 255)

        if click_selector.refPt is not None:
            point = click_selector.refPt[::-1]
            hls_point = hls_image[point]
            hls_min = np.min([hls_min, hls_point], axis=0)
            hls_max = np.max([hls_max, hls_point], axis=0)
            history.append([hls_min, hls_max])
            print('new hls_min', hls_min)
            print('new hls_max', hls_max)
            clone = paint_by_hls(hls_image, image, hls_min, hls_max)
            if old_clone is not None:
                old_clone = \
                    paint_by_hls(old_hls_image, old_image, hls_min, hls_max)

            click_selector.clear()

        if key == 'q':
            print('Quitting')
            break
        elif key == 'r':
            json.dump(
                {
                    'hls_min': hls_min.tolist(),
                    'hls_max': hls_max.tolist()
                },
                open(hls_range_file, 'w'))


if __name__ == '__main__':
    FLAGS(sys.argv)

    assert FLAGS.image_path
    assert exists(FLAGS.image_path)

    main()
