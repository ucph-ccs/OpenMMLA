def __getattr__(name):
    if name == "VLLMFrameAnalyzer":
        from .vllm_frame_analyzer import VLLMFrameAnalyzer
        return VLLMFrameAnalyzer
    if name == "FewShotVLLMFrameAnalyzer":
        from .few_shot_vllm_frame_analyzer import FewShotVLLMFrameAnalyzer
        return FewShotVLLMFrameAnalyzer
    if name == "ContextAwareVLLMFrameAnalyzer":
        from .context_aware_vllm_frame_analyzer import ContextAwareVLLMFrameAnalyzer
        return ContextAwareVLLMFrameAnalyzer
    if name == "MultiAngleVLLMFrameAnalyzer":
        from .multi_angle_vllm_frame_analyzer import MultiAngleVLLMFrameAnalyzer
        return MultiAngleVLLMFrameAnalyzer
    else:
        raise AttributeError(f"{name} not found")
