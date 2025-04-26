def __getattr__(name):
    if name == "VLLMFrameAnalyzer":
        from .vllm_frame_analyzer import VLLMFrameAnalyzer
        return VLLMFrameAnalyzer
    if name == "FewShotVLLMFrameAnalyzer":
        from .few_shot_vllm_frame_analyzer import FewShotVLLMFrameAnalyzer
        return FewShotVLLMFrameAnalyzer
    else:
        raise AttributeError(f"{name} not found")
