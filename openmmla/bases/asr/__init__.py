def __getattr__(name):
    if name == "ASRBase":
        from .asr_base import ASRBase
        return ASRBase
    elif name == 'ASRPostAnalyzer':
        from .asr_post_analyzer import ASRPostAnalyzer
        return ASRPostAnalyzer
    elif name == 'ASRSynchronizer':
        from .asr_synchronizer import ASRSynchronizer
        return ASRSynchronizer
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
