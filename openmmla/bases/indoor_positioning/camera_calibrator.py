import glob
import os

import cv2
import numpy as np
import yaml

from openmmla.bases import Base
from openmmla.utils.logger import get_logger
from .input import flush_input, get_function_calibrator


class CameraCalibrator(Base):
    """Camera calibration class for calibrating cameras with image capturing and calibration functions."""
    logger = get_logger('camera-calibrator')

    def __init__(self, project_dir: str, config_path: str):
        """Initialize the camera calibrator.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir, config_path)
        self._setup_from_yaml()
        self._setup_directories()

        """Calibration specific parameters."""
        self.CHECKERBOARD = (6, 9)  # Checkerboard dimensions
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Termination criteria

        """Runtime attributes."""
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)  # Object points array
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        self.obj_points = []  # Lists to store object points and image points
        self.img_points = []

    def _setup_from_yaml(self):
        """Set up attributes from YAML configuration."""
        image_config = self.config.get('Image', {})
        self.res_width = int(image_config.get('res_width', 0))
        self.res_height = int(image_config.get('res_height', 0))

    def _setup_directories(self):
        """Set up directories."""
        self.cameras_dir = os.path.join(self.project_dir, 'camera_calib/cameras')
        os.makedirs(self.cameras_dir, exist_ok=True)

    def run(self):
        """Run the camera calibrator."""
        func_map = {1: self._capture_images, 2: self._calibrate_camera}
        while True:
            try:
                select_fun = get_function_calibrator()
                if select_fun == 0:
                    self.logger.info("Exiting camera calibrator...")
                    break
                func_map.get(select_fun, lambda: print("Invalid option."))()
            except (Exception, KeyboardInterrupt) as e:
                self.logger.warning("%s, Come back to the main menu.", e, exc_info=True)

    def _capture_images(self):
        """Capture calibration images."""
        resolution = [self.res_width, self.res_height]
        flush_input()
        camera_name = input("Enter the camera name for capturing images: ")
        saved_directory = os.path.join(self.cameras_dir, camera_name)
        os.makedirs(saved_directory, exist_ok=True)

        available_seeds = self._detect_video_seeds()
        camera_seed = self._choose_camera_seed(available_seeds)
        if camera_seed is None:
            self.logger.warning("No available camera seed found.")
            return

        cam = cv2.VideoCapture(camera_seed)
        cam.set(3, resolution[0])
        cam.set(4, resolution[1])

        try:
            self.capture_and_save_image(cam, saved_directory)
        finally:
            cam.release()
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    def _detect_video_seeds(self):
        """Detect available video capture devices."""
        available_video_seeds = []
        number_of_detected_seeds = 0
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                print(f"{number_of_detected_seeds} : Camera seed {i} is available.")
                number_of_detected_seeds += 1
                available_video_seeds.append(i)
            cap.release()

        rtmp_config = self.config.get('RTMP', {})
        if 'video_streams' in rtmp_config:
            video_stream_list = rtmp_config['video_streams'].split(',')
            for streams in video_stream_list:
                if streams:
                    print(f"{number_of_detected_seeds} : RTMP stream {streams} is available.")
                    available_video_seeds.append(streams)
                    number_of_detected_seeds += 1

        return available_video_seeds

    def _choose_camera_seed(self, available_video_seeds):
        if not available_video_seeds:
            return None
        while True:
            try:
                seed_id = int(input("Choose your video seed id: "))
                if 0 <= seed_id < len(available_video_seeds):
                    return available_video_seeds[seed_id]
                else:
                    self.logger.warning("Invalid selection. Please choose a valid video seed.")
            except ValueError:
                self.logger.warning("Please enter a valid number.")

    def _calibrate_camera(self):
        """Calibrate the camera using a checkerboard pattern."""
        camera_name = self._select_camera()
        selected_path = os.path.join(self.cameras_dir, camera_name)
        is_fisheye = input("Is the camera a fisheye lens? (Y/n): ").lower() == 'y'

        try:
            images = glob.glob(os.path.join(selected_path, '*.jpg'))
            for img_file in images:
                img = cv2.imread(img_file)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                         cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
                if ret:
                    self.obj_points.append(self.objp)
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                                self.subpix_criteria) if not is_fisheye else corners
                    self.img_points.append(corners2)
                    cv2.drawChessboardCorners(img, self.CHECKERBOARD, corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(0)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

            if is_fisheye:
                ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                    self.obj_points, self.img_points, gray.shape[::-1], None, None,
                    flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND +
                           cv2.fisheye.CALIB_FIX_SKEW),
                    criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
            else:
                ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
                    self.obj_points, self.img_points, gray.shape[::-1], None, None)

            self._update_configuration(camera_name, K, D, is_fisheye)
        except Exception as e:
            self.logger.error("Error occurred during calibration: %s", e, exc_info=True)
        finally:
            self._clean_up()

    def _select_camera(self):
        """Select a camera folder."""
        directories = [d for d in os.listdir(self.cameras_dir) if os.path.isdir(os.path.join(self.cameras_dir, d))]
        for index, directory in enumerate(directories):
            print(f"{index}: {directory}")
        flush_input()
        choice = int(input("Enter the index of the camera folder you'd like to use: "))
        return directories[choice]

    def _clean_up(self):
        """Clean up object points and image points lists."""
        self.obj_points = []
        self.img_points = []

    def _update_configuration(self, camera_name, K, D, is_fisheye):
        """Update the configuration file with calibration results."""
        k_list = K.tolist()
        d_list = D.tolist()
        params = [float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])]

        # Create or update a camera section in config
        if 'Cameras' not in self.config:
            self.config['Cameras'] = {}
        if camera_name not in self.config['Cameras']:
            self.config['Cameras'][camera_name] = {}

        self.config['Cameras'][camera_name].update({
            'fisheye': is_fisheye,
            'params': params,
            'K': k_list,
            'D': d_list
        })

        # Write updated config to file
        with open(self.config_path, 'w') as config_file:
            yaml.safe_dump(self.config, config_file, sort_keys=False, default_flow_style=None)
        print(f"Configuration updated for {camera_name}.")

    @staticmethod
    def capture_and_save_image(cam, directory):
        image_number = 1
        print("Press 'c' to capture the image, or 'q' to quit.")
        while cam.isOpened():
            result, image = cam.read()
            if result:
                cv2.imshow("Real-Time Capture", image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    filename = f"{directory}/{image_number}.jpg"
                    cv2.imwrite(filename, image)
                    print(f"Image captured as {filename}")
                    cv2.imshow("Captured Image", image)
                    cv2.waitKey(1)
                    flush_input()
                    user_input = input("Are you satisfied with the image? (Y/n): ")
                    if user_input.lower() == 'y':
                        image_number += 1
                        print(f"Image saved as {filename}")
                    else:
                        print("Image discarded. Continue capturing...")
                        os.remove(filename)
                    cv2.destroyWindow("Captured Image")
                elif key == ord('q'):
                    return
            else:
                print("No image detected. Please try again.")
                return
