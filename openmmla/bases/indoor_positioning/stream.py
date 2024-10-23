import gc
import os
import threading
import time

import cv2

from openmmla.utils.logger import get_logger


class WebcamVideoStream:
    logger = get_logger('WebcamVideoStream')

    def __init__(self, format, res, src=0, save_path=''):
        self.stream = cv2.VideoCapture(src)
        self.stream.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*format))
        self.stream.set(3, res[0])  # Width
        self.stream.set(4, res[1])  # Height
        self.stream.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.grabbed, self.frame = self.stream.read()
        self.save_path = save_path

        # Determine the type of save files, either video or images
        if self.save_path:
            if any(self.save_path.lower().endswith(ext) for ext in ['.avi', '.mp4']):
                fourcc = cv2.VideoWriter_fourcc(*format)
                self.video_writer = cv2.VideoWriter(self.save_path, fourcc, 30.0, res)
            else:
                if not os.path.exists(self.save_path):
                    os.makedirs(self.save_path)

        self.stop_event = threading.Event()
        self.threads = []

    def start(self):
        self.stop_event.clear()
        self._start_thread(self._update)
        return self

    def stop(self):
        self.stop_event.set()
        for t in self.threads:
            if threading.current_thread() != t:
                try:
                    t.join(timeout=5)
                except Exception as e:
                    self.logger.warning("During the thread stopping, catch exception %s", e, exc_info=True)
        self.threads.clear()
        self._free_memory()

    def read(self):
        return self.frame

    def _start_thread(self, target, *args):
        t = threading.Thread(target=target, args=args)
        t.daemon = True
        self.threads.append(t)
        t.start()

    def _update(self):
        start_time = time.time()
        fps = 0
        frame_id = 0
        frame_count = 0

        while not self.stop_event.is_set():
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                raise Exception('Frame not grabbed')

            # Display fps on frame
            frame_count += 1
            if frame_count % 30 == 0:
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time
                start_time = time.time()
                frame_count = 0
                print(f'Webcam FPS: {fps:.2f}')  # Uncomment to print fps for testing
            # cv2.putText(self.frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
            #             2)

            if self.save_path:
                if hasattr(self, 'video_writer'):
                    self.video_writer.write(self.frame)
                else:
                    saved_frame = cv2.resize(self.frame, (960, 540))
                    cv2.imwrite(os.path.join(self.save_path, f'frame_{frame_id:07d}.jpg'), saved_frame)
                    frame_id += 1

    def _free_memory(self):
        if hasattr(self, 'video_writer'):
            self.video_writer.release()
        self.stream.release()
        gc.collect()

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc_value, traceback):
        self.stop()
