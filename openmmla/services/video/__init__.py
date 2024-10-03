def __getattr__(name):
    if name == "VideoFrameAnalyzer":
        from .video_frame_analyzer import VideoFrameAnalyzer
        return VideoFrameAnalyzer
    else:
        raise AttributeError(f"{name} not found")