def __getattr__(name):
    if name == "VFABase":
        from .vfa_base import VFABase
        return VFABase
