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


def get_stream_urls(config: dict, protocol=None) -> list[str]:
    """extract the URLs bases can pull from the Streams config section, in config order.

    An entry is pulled from its read_target (e.g. the RTSP URL MediaMTX serves)
    when set, otherwise from its target. Only rtmp/rtsp/srt URLs qualify;
    ``protocol`` (one scheme or a tuple of schemes) narrows them. Falls back to
    the legacy RTMP section (audio_streams / video_streams) if Streams is empty.
    """
    prefixes = _scheme_prefixes(protocol)
    streams = config.get("Streams", {})
    if isinstance(streams, dict) and streams:
        urls = []
        for entry in streams.values():
            if isinstance(entry, dict):
                url = stream_read_url(entry)
                if url.startswith(prefixes):
                    urls.append(url)
        if urls:
            return urls

    rtmp = config.get("RTMP", {})
    if isinstance(rtmp, dict):
        for key in ("audio_streams", "video_streams"):
            val = rtmp.get(key)
            if isinstance(val, list):
                return [u for u in val if isinstance(u, str) and u.startswith(prefixes)]
            if isinstance(val, str) and "," in val:
                return [u.strip() for u in val.split(",") if u.strip().startswith(prefixes)]
    return []