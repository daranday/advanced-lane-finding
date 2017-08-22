import argparse
import pickle
from os.path import join, splitext, basename, exists, abspath

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, ImageSequenceClip
import matplotlib.pyplot as plt

from calibrate import CameraCalibrator
from warp import PerspectiveTransformFinder
from utils import (PROJECT_DIR, OUTPUT_DIR, CALIBRATION_DIR, create_dirs,
                   OpencvClicker)

import logging
log_format = '[%(asctime)s][%(name)s][%(levelname)s] %(message)s'
logging.basicConfig(level=logging.DEBUG, format=log_format)
logger = logging.getLogger(__name__)


_lane_detection_debug_dir = join(OUTPUT_DIR, 'lane_detection')

parser = argparse.ArgumentParser()
parser.add_argument('--video_path', help='Video for lane line detection',
                    required=True)
parser.add_argument('--straight_lane_img_file',
                    default=join(
                        PROJECT_DIR, 'test_images/straight_lines1.jpg'),
                    help='Straight lane image file')
args = None

RED = (0, 0, 255)

plt.ion()
plt.show()
colorbared = False


class ImageThresholder(object):
    """docstring for ImageThresholder"""

    def __init__(self):
        self.s_threshold = (109, 255)
        self.sobel_x_threshold = (50, 100)

    def process(self, img):
        img = img.copy()
        # Convert to HSV color space and separate the V channel
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
        l_channel = hsv[:, :, 1]
        s_channel = hsv[:, :, 2]

        # Sobel x
        # Take the derivative in x
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)
        # Absolute x derivative to accentuate lines away from horizontal
        abs_sobelx = np.absolute(sobelx)
        scaled_sobel = np.uint8(255 * abs_sobelx / np.max(abs_sobelx))

        # Threshold x gradient
        sxbinary = np.zeros_like(scaled_sobel, dtype=np.uint8)
        sxbinary[(scaled_sobel >= self.sobel_x_threshold[0]) &
                 (scaled_sobel <= self.sobel_x_threshold[1])] = 1

        # Threshold color channel
        s_binary = np.zeros_like(s_channel, dtype=np.uint8)
        s_binary[(s_channel >= self.s_threshold[0]) &
                 (s_channel <= self.s_threshold[1])] = 1

        # Stack each channel
        # Note color_binary[:, :, 0] is all 0s, effectively an all black image.
        # It might be beneficial to replace this channel with something else.
        # color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary))
        color_binary = sxbinary | s_binary
        return color_binary


class LaneLineFinder(object):
    """docstring for LaneLineFinder"""

    def __init__(self):
        self.leftx_base = None
        self.rightx_base = None
        self.debug_img = None

    def initialize_upward_search(self, binary_img):
        """Find starting points from peaks of bottom histogram."""
        h, w = binary_img.shape
        bottom_histogram = np.sum(binary_img[h // 2:, :], axis=0)
        midpoint = np.int(bottom_histogram.shape[0] / 2)
        self.leftx_base = np.argmax(bottom_histogram[:midpoint])
        self.rightx_base = np.argmax(bottom_histogram[midpoint:]) + midpoint
        self.debug_img = binary_img[..., np.newaxis].repeat(3, axis=2) * 255

    def find(self, binary_warped, M, undist):
        # Begin by choosing two starting points from lower half of image.
        self.initialize_upward_search(binary_warped)
        # Choose the number of sliding windows
        nwindows = 9
        # Set height of windows
        window_height = np.int(binary_warped.shape[0] / nwindows)
        # Identify the x and y positions of all nonzero pixels in the image
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Current positions to be updated for each window
        leftx_current = self.leftx_base
        rightx_current = self.rightx_base
        # Set the width of the windows +/- margin
        margin = 100
        # Set minimum number of pixels found to recenter window
        minpix = 50
        # Create empty lists to receive left and right lane pixel indices
        left_lane_inds = []
        right_lane_inds = []

        # Step through the windows one by one
        for window in range(nwindows):
            # Identify window boundaries in x and y (and right and left)
            win_y_low = binary_warped.shape[0] - (window + 1) * window_height
            win_y_high = binary_warped.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin
            # Draw the windows on the visualization image
            cv2.rectangle(self.debug_img, (win_xleft_low, win_y_low),
                          (win_xleft_high, win_y_high), (0, 255, 0), 2)
            cv2.rectangle(self.debug_img, (win_xright_low, win_y_low),
                          (win_xright_high, win_y_high), (0, 255, 0), 2)
            # Identify the nonzero pixels in x and y within the window
            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (
                nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
            # Append these indices to the lists
            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)
            # If you found > minpix pixels, recenter next window on their mean
            # position
            if len(good_left_inds) > minpix:
                leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

        # Concatenate the arrays of indices
        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)
        if len(right_lane_inds) < 500:
            raise Exception()

        # Extract left and right line pixel positions
        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Fit a second order polynomial to each
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # Generate x and y values for plotting
        ploty = np.linspace(0, binary_warped.shape[
                            0] - 1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + \
            right_fit[1] * ploty + right_fit[2]

        self.debug_img[lefty, leftx] = [255, 0, 0]
        self.debug_img[righty, rightx] = [0, 0, 255]

        # Create an image to draw the lines on
        warp_zero = np.zeros_like(binary_warped).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        # Recast the x and y points into usable format for cv2.fillPoly()
        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array(
            [np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        # Draw the lane onto the warped blank image
        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))

        # Warp the blank back to original image space using inverse perspective
        # matrix (Minv)
        newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(M),
                                      (warp_zero.shape[1], warp_zero.shape[0]))
        # Combine the result with the original image
        lane_area_img = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)

        y_m_per_pixel = 3. / (272 - 63)
        x_m_per_pixel = 3.7 / (2238 - 1600)
        left_fit = np.polyfit(lefty * y_m_per_pixel,
                              leftx * x_m_per_pixel, 2)
        right_fit = np.polyfit(righty * y_m_per_pixel,
                               rightx * x_m_per_pixel, 2)
        logger.info('left_fit {}'.format(left_fit))
        logger.info('right_fit {}'.format(right_fit))
        # Generate x and y values for plotting
        y_eval = binary_warped.shape[0]
        left_curverad = (
            (1 + (2*left_fit[0]*y_eval*y_m_per_pixel +
                  left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
        right_curverad = (
            (1 + (2*right_fit[0]*y_eval*y_m_per_pixel +
                  right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
        if right_curverad < 100:
            raise Exception()

        cv2.putText(lane_area_img, 'Left Curvature Radius: {:.2f} m'.format(
            left_curverad), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 1)
        cv2.putText(lane_area_img, 'Right Curvature Radius: {:.2f} m'.format(
            right_curverad), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.0, RED, 1)

        lane_area_img = lane_area_img[..., ::-1]
        return lane_area_img


class LaneDetector(object):
    """docstring for LaneDetector"""

    def __init__(self, debug_dir=_lane_detection_debug_dir):
        self.debug_dir = debug_dir
        self.calibrator = CameraCalibrator()
        self.calibrator.calibrate(CALIBRATION_DIR)
        self.thresholder = ImageThresholder()
        self.lane_line_finder = LaneLineFinder()
        self.output_frames = []
        self.find_warping_params()
        create_dirs([self.debug_dir])

    def find_warping_params(self):
        straight_lane_img = cv2.imread(args.straight_lane_img_file)
        finder = PerspectiveTransformFinder()
        self.warper = finder.find(straight_lane_img)

    def process_frame(self, frame_i, rgb_frame):
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
        frame = self.calibrator.undistort(frame)
        binary_image = self.thresholder.process(frame)
        binary_gray = cv2.cvtColor(binary_image * 255, cv2.COLOR_GRAY2BGR)
        gray_warped = self.warper.warp(binary_gray)
        binary_warped = self.warper.warp(binary_image)
        try:
            self.lanes_img = self.lane_line_finder.find(
                binary_warped, self.warper.M, rgb_frame)
        except:
            save_path = join(self.debug_dir, 'frame_{}.jpg'.format(frame_i))
            cv2.imwrite(save_path, frame)
            logger.error('Lane finding failed.')
            return

        # canvas = np.vstack([
        #     np.hstack([frame, self.lanes_img]),
            # np.hstack([gray_warped, lanes_img]),
        # ])
        canvas = self.lanes_img
        cv2.imshow(basename(self.video_path), canvas)
        self.output_frames.append(canvas[..., ::-1])

        key = cv2.waitKey(1 - self.paused)
        button = chr(key & 0xff)
        if button == 'q':
            exit(0)
        elif button == 'p':
            self.paused = 1 - self.paused
        elif button == 's':
            save_path = join(self.debug_dir, 'frame_{}.jpg'.format(frame_i))
            cv2.imwrite(save_path, frame)
            logger.info('[i] Successfully saved frame to {}'.format(save_path))

    def process_video(self, video_path):
        self.paused = 0
        self.video_path = video_path
        logger.info(video_path)
        clip = VideoFileClip(video_path)
        for i, rgb_frame in enumerate(clip.iter_frames()):
            self.process_frame(i, rgb_frame)

    def detect(self, video_path):
        try:
            self.output_frames = np.load('output_frames.npy')
            self.output_frames = [self.output_frames[i] for i in range(self.output_frames.shape[0])]
        except:
            self.process_video(video_path)
            np.save('output_frames.npy', self.output_frames)
        video = ImageSequenceClip(self.output_frames, fps=20)
        video.write_videofile(join(OUTPUT_DIR, 'output.avi'), fps=20,
                              codec='png'
                              # bitrate='64k',
                              )


if __name__ == '__main__':
    args = parser.parse_args()
    args.video_path = abspath(args.video_path)
    assert exists(args.video_path)

    lane_detector = LaneDetector()
    lane_detector.detect(args.video_path)
