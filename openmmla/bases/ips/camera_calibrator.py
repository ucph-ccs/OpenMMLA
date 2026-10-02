import glob
import os

import cv2
import numpy as np

from openmmla.bases.base import Base
from openmmla.utils.input import show_error_and_pause
from openmmla.utils.logger import get_logger
from openmmla.utils.video.turn import normalize_turn, stream_capture_turn
from openmmla.utils.yaml_dump import dump_yaml_pretty
from .input import flush_input, get_function_calibrator

# what undoes a clockwise turn of a picture, by the turn: cv2.rotate's code for the turn back
TURN_BACK = {90: cv2.ROTATE_90_COUNTERCLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_CLOCKWISE}


def sensor_picture(image, turn):
    """`image` as the sensor gave it, of a picture turned `turn` degrees clockwise on its way (a
    stream whose Streams entry has a rotate, openmmla.utils.video.turn); the image itself when it
    was not turned."""
    turn = normalize_turn(turn)
    return cv2.rotate(image, TURN_BACK[turn]) if turn else image


class CameraCalibrator(Base):
    """Class for calibrating cameras with image capturing and calibration functions.

    A Cameras entry holds the intrinsics of the picture as the sensor gives it, which the bases turn
    with the picture (openmmla.utils.video.turn). A stream turned where it is captured (its Streams
    entry's rotate) sends turned pictures, so each image captured from it is turned back before it
    is saved: the checkerboards, the intrinsics and calibration_resolution are the sensor's whatever
    turned the stream. The live view shows the picture as it comes, turned."""
    logger = get_logger('camera-calibrator')

    def __init__(self, project_dir: str | None, config_path: str):
        """Initialize the camera calibrator.

        Args:
            project_dir: path to the project directory
            config_path: path to the configuration file
        """
        super().__init__(project_dir=project_dir, config_path=config_path)

        """Calibration specific parameters."""
        self.CHECKERBOARD = (6, 9)  # Checkerboard dimensions
        self.subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)  # Termination criteria

        """Runtime attributes."""
        self.objp = np.zeros((1, self.CHECKERBOARD[0] * self.CHECKERBOARD[1], 3), np.float32)  # Object points array
        self.objp[0, :, :2] = np.mgrid[0:self.CHECKERBOARD[0], 0:self.CHECKERBOARD[1]].T.reshape(-1, 2)
        self.obj_points = []  # Lists to store object points and image points
        self.img_points = []
        # {stream url: the turn its capture applies}, of the streams offered as video seeds
        self.seed_turns: dict = {}

        self._setup_from_yaml()
        self._setup_directories()

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
                if not isinstance(e, KeyboardInterrupt):
                    show_error_and_pause(e, "return to the Camera Calibrator menu")

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
        turn = self.seed_turns.get(camera_seed, 0)
        if turn:
            print(f"The stream is turned {turn} degrees where it is captured: each image is turned back "
                  f"before it is saved, so the intrinsics are the sensor's.")

        try:
            self.capture_and_save_image(cam, saved_directory, turn=turn)
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

        from openmmla.utils.constants import get_stream_sources
        for name, url in get_stream_sources(self.config):
            turn = stream_capture_turn(self.config, url, name)
            self.seed_turns[url] = turn
            turned = f" (turned {turn} degrees where it is captured)" if turn else ""
            print(f"{number_of_detected_seeds} : Stream {url} is available{turned}.")
            available_video_seeds.append(url)
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
            images = sorted(glob.glob(os.path.join(selected_path, '*.jpg')))
            K, D, size = self.calibrate_images(images, is_fisheye, show=True)
            self._update_configuration(camera_name, K, D, is_fisheye, size)
        except Exception as e:
            self.logger.error("Error occurred during calibration: %s", e, exc_info=True)
        finally:
            self._clean_up()

    def calibrate_images(self, image_files, is_fisheye=False, show=False):
        """(K, D, (width, height)) of the checkerboard images `image_files`, the size the
        intrinsics hold for. The images are taken as saved, as the sensor gave them; those of
        another size than most of them (one captured turned a quarter before ips-ccal turned the
        images back, or at another resolution) are left out with a warning, since one calibration
        holds for one frame size."""
        found = []
        for img_file in image_files:
            img = cv2.imread(img_file)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, self.CHECKERBOARD,
                                                     cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
            if ret:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                            self.subpix_criteria) if not is_fisheye else corners
                found.append((img_file, gray.shape[::-1], corners2))
                if show:
                    cv2.drawChessboardCorners(img, self.CHECKERBOARD, corners2, ret)
                    cv2.imshow('img', img)
                    cv2.waitKey(0)
        if show:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        if not found:
            raise ValueError("No checkerboard was found on the images.")
        sizes = [size for _, size, _ in found]
        size = max(sorted(set(sizes)), key=sizes.count)
        others = [os.path.basename(img_file) for img_file, other, _ in found if other != size]
        if others:
            self.logger.warning("Left out %d image(s) of another size than %dx%d: %s", len(others), size[0], size[1],
                                ", ".join(others))
        self.obj_points = [self.objp for _, other, _ in found if other == size]
        self.img_points = [corners for _, other, corners in found if other == size]
        if is_fisheye:
            ret, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                self.obj_points, self.img_points, size, None, None,
                flags=(cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND +
                       cv2.fisheye.CALIB_FIX_SKEW),
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6))
        else:
            ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
                self.obj_points, self.img_points, size, None, None)
        return K, D, size

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

    def _update_configuration(self, camera_name, K, D, is_fisheye, image_size=None):
        """Update the configuration file with calibration results.

        image_size is the (width, height) of the checkerboard images: the frame size the intrinsics
        hold for, kept as the camera's calibration_resolution so that the IPS base scales them to
        frames of another size."""
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
        if image_size is not None:
            self.config['Cameras'][camera_name]['calibration_resolution'] = [int(image_size[0]), int(image_size[1])]

        # Write updated config to file
        dump_yaml_pretty(self.config, self.config_path)
        print(f"Configuration updated for {camera_name}.")

    @staticmethod
    def next_image_number(directory) -> int:
        """the number after the highest <n>.jpg already there: a second capture
        for a camera adds to its images instead of writing over 1.jpg, 2.jpg ..."""
        try:
            stems = [os.path.splitext(name)[0] for name in os.listdir(directory)]
        except OSError:
            return 1
        return max((int(stem) for stem in stems if stem.isdigit()), default=0) + 1

    @staticmethod
    def capture_and_save_image(cam, directory, turn=0):
        """capture images on 'c', saved as the sensor gave them: an image of a picture turned `turn`
        degrees on its way is turned back first."""
        image_number = CameraCalibrator.next_image_number(directory)
        print("Press 'c' to capture the image, or 'q' to quit.")
        while cam.isOpened():
            result, image = cam.read()
            if result:
                cv2.imshow("Real-Time Capture", image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('c'):
                    filename = f"{directory}/{image_number}.jpg"
                    cv2.imwrite(filename, sensor_picture(image, turn))
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
