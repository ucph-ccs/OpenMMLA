def __getattr__(name):
    if name == "MultiAngleVLLMFrameAnalyzer":
        from .multi_angle_vllm_frame_analyzer import MultiAngleVLLMFrameAnalyzer
        return MultiAngleVLLMFrameAnalyzer
    else:
        raise AttributeError(f"{name} not found")
