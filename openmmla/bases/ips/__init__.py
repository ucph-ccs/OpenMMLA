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
    elif name == 'IPSBase':
        from .ips_base import IPSBase
        return IPSBase
    elif name == 'IPSSynchronizer':
        from .ips_synchronizer import IPSSynchronizer
        return IPSSynchronizer
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
