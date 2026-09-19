# influxdb measurement and bucket defaults
INFLUXDB_MEASUREMENT = "sensor_events"
INFLUXDB_DEFAULT_BUCKET = "mmla-data"

# event type constants for influxdb tags
EVENT_TYPE_ASR_TRANSCRIPTION = "asr_transcription"
EVENT_TYPE_ASR_RECOGNITION = "asr_recognition"
EVENT_TYPE_IPS_TRANSLATION = "ips_translation"
EVENT_TYPE_IPS_ROTATION = "ips_rotation"
EVENT_TYPE_IPS_RELATION = "ips_relation"
EVENT_TYPE_VFA_ACTION = "vfa_action"
EVENT_TYPE_PARTICIPANT_INDICATORS = "participant_indicators"
EVENT_TYPE_PARTICIPANT_SUMMARY = "participant_summary"
EVENT_TYPE_GROUP_INDICATORS = "group_indicators"
EVENT_TYPE_GROUP_SUMMARY = "group_summary"

# grouped by pipeline for management operations
EVENT_TYPES_ASR = [EVENT_TYPE_ASR_TRANSCRIPTION, EVENT_TYPE_ASR_RECOGNITION]
EVENT_TYPES_IPS = [EVENT_TYPE_IPS_TRANSLATION, EVENT_TYPE_IPS_ROTATION, EVENT_TYPE_IPS_RELATION]
EVENT_TYPES_VFA = [EVENT_TYPE_VFA_ACTION]
EVENT_TYPES_ANALYTICS = [
    EVENT_TYPE_PARTICIPANT_INDICATORS,
    EVENT_TYPE_PARTICIPANT_SUMMARY,
    EVENT_TYPE_GROUP_INDICATORS,
    EVENT_TYPE_GROUP_SUMMARY,
]
EVENT_TYPES_ALL = EVENT_TYPES_ASR + EVENT_TYPES_IPS + EVENT_TYPES_VFA + EVENT_TYPES_ANALYTICS

# mongodb defaults
MONGODB_DEFAULT_DB = "openmmla"


# URL schemes a base can pull a live stream from; udp/tcp targets are audio
# pushed straight to an ASR base and are never pulled
STREAM_URL_SCHEMES = ("rtmp", "rtsp", "srt")

# 'rtmp' was the name of the URL source before MediaMTX; both spell the same thing
STREAM_SOURCE_ALIASES = {"rtmp": "stream"}


def normalize_source(source) -> str:
    """canonical source name of a base ('rtmp' is an alias of 'stream')."""
    text = str(source or "").strip().lower()
    return STREAM_SOURCE_ALIASES.get(text, text)


# device names that capture sound: ALSA's, and a Mac's microphone alone
# (":<index>" to AVFoundation)
AUDIO_DEVICE_PREFIXES = ("hw:", "plughw:", "default", "sysdefault", "dsnoop", "plug:", "pulse", ":", "none:")


def _entry_field(entry, name: str) -> str:
    value = entry.get(name) if isinstance(entry, dict) else getattr(entry, name, "")
    return str(value or "").strip()


def stream_kind(entry, default: str = "video") -> str:
    """'audio' or 'video' of a Streams entry (a dict, or anything with kind,
    target and device): its kind when set; else a udp/tcp target (raw audio to
    an ASR base) or a device that captures sound; else the default, which is
    the card's: a microphone pushed over RTMP from a Mac names no device (its
    first one is ''), so only the pipeline that pulls it can tell. The Streams
    tab records by this, and a session's sources look the recording up by it."""
    kind = _entry_field(entry, "kind").lower()
    if kind in ("audio", "video"):
        return kind
    if _entry_field(entry, "target").startswith(("udp://", "tcp://")):
        return "audio"
    if _entry_field(entry, "device").lower().startswith(AUDIO_DEVICE_PREFIXES):
        return "audio"
    return default if default in ("audio", "video") else "video"


def stream_read_url(entry: dict) -> str:
    """the URL a base pulls a Streams entry from: read_target when set, else target."""
    read_target = str(entry.get("read_target") or "").strip()
    return read_target or str(entry.get("target") or "").strip()


def _scheme_prefixes(protocol) -> tuple[str, ...]:
    if protocol is None:
        schemes = STREAM_URL_SCHEMES
    elif isinstance(protocol, str):
        schemes = (protocol,)
    else:
        schemes = tuple(protocol)
    return tuple(f"{scheme}://" for scheme in schemes)


def get_stream_sources(config: dict, protocol=None) -> list[tuple[str, str]]:
    """(name, url) of the streams bases can pull from the Streams config
    section, in config order: what a base's source_index counts through when
    its source is 'stream'.

    An entry is pulled from its read_target (e.g. the RTSP URL MediaMTX serves)
    when set, otherwise from its target. Only rtmp/rtsp/srt URLs qualify;
    ``protocol`` (one scheme or a tuple of schemes) narrows them. Falls back to
    the legacy RTMP section (audio_streams / video_streams) if Streams is empty,
    whose URLs are named after their last path segment.
    """
    prefixes = _scheme_prefixes(protocol)
    streams = config.get("Streams", {})
    if isinstance(streams, dict) and streams:
        found = []
        for name, entry in streams.items():
            if isinstance(entry, dict):
                url = stream_read_url(entry)
                if url.startswith(prefixes):
                    found.append((str(name), url))
        if found:
            return found

    rtmp = config.get("RTMP", {})
    if isinstance(rtmp, dict):
        for key in ("audio_streams", "video_streams"):
            val = rtmp.get(key)
            if isinstance(val, list):
                urls = [u for u in val if isinstance(u, str) and u.startswith(prefixes)]
            elif isinstance(val, str) and "," in val:
                urls = [u.strip() for u in val.split(",") if u.strip().startswith(prefixes)]
            else:
                continue
            return [(u.rstrip("/").rsplit("/", 1)[-1], u) for u in urls]
    return []


def get_stream_urls(config: dict, protocol=None) -> list[str]:
    """the URLs bases can pull, in config order (see get_stream_sources)."""
    return [url for _, url in get_stream_sources(config, protocol)]


def resolve_stream_source(config: dict, source_index) -> tuple[str, str]:
    """(name, url) of the stream a base with source: stream pulls.

    source_index names a Streams entry, which is what the console writes. A
    number is read as the position among the pullable entries, as older
    configs have it, and a URL or its last path segment is matched too.
    Anything else raises ValueError naming the streams there are: a base used
    to take the first one then, and quietly looked through the wrong camera."""
    sources = get_stream_sources(config)
    if not sources:
        raise ValueError("No pullable stream (rtmp/rtsp/srt URL) found in the Streams config section.")
    names = ", ".join(name for name, _ in sources)
    text = "" if source_index is None else str(source_index).strip()
    if not text:
        if len(sources) == 1:
            return sources[0]
        raise ValueError(
            f"The base names no stream in source_index, and there are {len(sources)} to pull: {names}.")
    for name, url in sources:
        if text == name:
            return name, url
    if text.isdigit():
        position = int(text)
        if position < len(sources):
            return sources[position]
        raise ValueError(
            f"source_index {position} is past the {len(sources)} pullable stream(s): {names}.")
    for name, url in sources:
        if text == url or text == url.rstrip("/").rsplit("/", 1)[-1]:
            return name, url
    raise ValueError(f"source_index '{text}' is none of the pullable streams: {names}.")