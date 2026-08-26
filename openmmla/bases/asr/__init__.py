def __getattr__(name):
    if name == "ASRBase":
        from .asr_base import ASRBase
        return ASRBase
    elif name == 'ASRSynchronizer':
        from .asr_synchronizer import ASRSynchronizer
        return ASRSynchronizer
    elif name == 'start_asr_base':
        from .asr_base import start_asr_base
        return start_asr_base
    elif name == 'start_asr_synchronizer':
        from .asr_synchronizer import start_asr_synchronizer
        return start_asr_synchronizer
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
