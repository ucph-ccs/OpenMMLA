def __getattr__(name):
    if name == "SpeechEnhancer":
        from .speech_enhancer import SpeechEnhancer
        return SpeechEnhancer
    elif name == "AudioInferer":
        from .audio_inferer import AudioInferer
        return AudioInferer
    elif name == "AudioResampler":
        from .audio_resampler import AudioResampler
        return AudioResampler
    elif name == "SpeechSeparator":
        from .speech_separator import SpeechSeparator
        return SpeechSeparator
    elif name == "SpeechTranscriber":
        from .speech_transcriber import SpeechTranscriber
        return SpeechTranscriber
    elif name == "VoiceActivityDetector":
        from .voice_activity_detector import VoiceActivityDetector
        return VoiceActivityDetector
    else:
        raise AttributeError(f"{name} not found")
