def __getattr__(name):
    if name == "CameraCalibrator":
        from .camera_calibrator import CameraCalibrator
        return CameraCalibrator
    elif name == 'CameraSyncManager':
        from .camera_sync_manager import CameraSyncManager
        return CameraSyncManager
    elif name == 'CameraTagDetector':
        from .camera_tag_detector import CameraTagDetector
        return CameraTagDetector
    elif name == 'VideoBase':
        from .video_base import VideoBase
        return VideoBase
    elif name == 'VideoSynchronizer':
        from .video_synchronizer import VideoSynchronizer
        return VideoSynchronizer
    elif name == 'VideoVisualizer':
        from .video_visualizer import VideoVisualizer
        return VideoVisualizer
    elif name == 'WebcamVideoStream':
        from .stream import WebcamVideoStream
        return WebcamVideoStream
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
