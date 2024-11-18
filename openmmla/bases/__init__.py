def __getattr__(name):
    if name == "Base":
        from .base import Base
        return Base
    elif name == 'Synchronizer':
        from .synchronizer import Synchronizer
        return Synchronizer
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
