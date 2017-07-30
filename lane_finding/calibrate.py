import pickle
from glob import glob
from os.path import join, basename

import cv2
import numpy as np

from utils import CALIBRATION_DIR, OUTPUT_DIR, create_dirs


_calibration_debug_dir = join(OUTPUT_DIR, 'calibration')


class CameraCalibrator(object):
    """docstring for CameraCalibrator"""

    def __init__(self, debug_dir=_calibration_debug_dir):
        self.debug_dir = debug_dir
        self.calib_params_file = join(debug_dir, 'calibration_params.p')
        self.calibration_dir = None
        self.calib_params = None
        self.nx = 9
        self.ny = 6

    def load(self):
        self.calib_params = pickle.load(open(self.calib_params_file, 'rb'))
        print('[i] Loading camera calibration params from',
              self.calib_params_file)

    def save(self):
        pickle.dump(self.calib_params, open(self.calib_params_file, 'wb'))
        print('[i] Saving camera calibration params to',
              self.calib_params_file)

    def calibrate(self, calibration_dir):
        create_dirs([self.debug_dir])
        self.calibration_dir = calibration_dir

        # Load saved calibration parameters.
        try:
            self.load()
            return
        except:
            pass

        # Get calibration images.
        img_paths = sorted(glob(join(self.calibration_dir, '*')))

        # Prepare object points.
        objpoint = np.zeros((self.nx * self.ny, 3), np.float32)
        objpoint[:, :2] = np.mgrid[0:self.ny, 0:self.nx].T.reshape(-1, 2)

        # Need to match 3D coordinates to 2D image coordinates.
        objpoints = []  # 3d coordinates.
        imgpoints = []  # 2d coordinates.

        # Corner extraction termination criteria.
        criteria = (cv2.TERM_CRITERIA_EPS +
                    cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        corners_dims = (self.ny, self.nx)
        for img_path in img_paths:
            img = cv2.imread(img_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, corners_dims, None)

            # If found, add object points, image points (after refining them)
            if ret is True:
                corners = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), criteria)
                imgpoints.append(corners)
                objpoints.append(objpoint)

                # Draw and display the corners
                img = cv2.drawChessboardCorners(img, corners_dims, corners, ret)

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None)
        self.calib_params = {'mtx': mtx, 'dist': dist}
        self.save()

        print('[d] Successfully calibrated camera params to', self.debug_dir)

    def undistort(self, img):
        # Undistort.
        undistorted_img = cv2.undistort(img,
                                        self.calib_params['mtx'],
                                        self.calib_params['dist'],
                                        None,
                                        None)
        return undistorted_img


class CameraCalibratorTest(object):

    def undistort_images(self, input_dir, output_dir):
        img_paths = sorted(glob(join(input_dir, '*')))
        for img_path in img_paths:
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            undistorted_img = self.calibrator.undistort(img)
            debug_img_path = join(output_dir, basename(img_path))
            cv2.imwrite(debug_img_path, undistorted_img)

    def main(self):
        self.calibrator = CameraCalibrator()
        self.calibrator.calibrate(CALIBRATION_DIR)
        self.undistort_images(CALIBRATION_DIR, self.calibrator.debug_dir)


if __name__ == '__main__':
    test = CameraCalibratorTest()
    test.main()
