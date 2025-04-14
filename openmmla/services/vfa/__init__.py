def __getattr__(name):
    if name == "VLLMFrameAnalyzer":
        from .vllm_frame_analyzer import VLLMFrameAnalyzer
        return VLLMFrameAnalyzer
    else:
        raise AttributeError(f"{name} not found")
