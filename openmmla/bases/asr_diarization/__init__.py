def __getattr__(name):
    if name == "AudioBase":
        from .audio_base import AudioBase
        return AudioBase
    elif name == 'PostAudioAnalyzer':
        from .post_audio_analyzer import PostAudioAnalyzer
        return PostAudioAnalyzer
    elif name == 'AudioSynchronizer':
        from .audio_synchronizer import AudioSynchronizer
        return AudioSynchronizer
    elif name == 'BadgeAudioBase':
        from .badge_audio_base import BadgeAudioBase
        return BadgeAudioBase
    elif name == 'JabraAudioBase':
        from .jabra_audio_base import JabraAudioBase
        return JabraAudioBase
    else:
        raise AttributeError(f"module {__name__} has no attribute {name}")
