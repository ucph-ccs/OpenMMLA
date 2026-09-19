"""the language a base asks the speech transcriber for, in the form each backend takes: whisper,
WhisperX and DashScope a language code (en), Azure a locale (en-US). A base may send either."""

# the locale Azure transcribes a language code in, where it is not <code>-<CODE>
_AZURE_LOCALES = {
    "ar": "ar-SA", "cs": "cs-CZ", "da": "da-DK", "el": "el-GR", "en": "en-US", "he": "he-IL",
    "hi": "hi-IN", "ja": "ja-JP", "ko": "ko-KR", "nb": "nb-NO", "no": "nb-NO", "sv": "sv-SE",
    "uk": "uk-UA", "vi": "vi-VN", "zh": "zh-CN",
}


def language_code(language: str | None) -> str | None:
    """the language code of `language`: a locale's part before its region (en-US: en). None for
    none, and for the 'null' a config writes for auto-detection."""
    text = str(language or "").strip()
    if not text or text.lower() == "null":
        return None
    return text.split("-")[0].lower() or None


def azure_locale(language: str | None) -> str | None:
    """the locale Azure transcribes `language` in: a locale as it is (en-GB), a code as Azure names
    it (en: en-US, de: de-DE); None for none."""
    text = str(language or "").strip()
    if not text:
        return None
    if "-" in text:
        return text
    code = text.lower()
    return _AZURE_LOCALES.get(code, f"{code}-{code.upper()}")
