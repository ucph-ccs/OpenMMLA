def __getattr__(name):
    if name == "VFABase":
        from .vfa_base import VFABase
        return VFABase
    elif name == "VFASynchronizer":
        from .vfa_synchronizer import VFASynchronizer
        return VFASynchronizer
    raise AttributeError(f"module 'openmmla.bases.vfa' has no attribute '{name}'")
