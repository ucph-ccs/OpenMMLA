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


def get_stream_urls(config: dict, protocol: str = "rtmp") -> list[str]:
    """extract target URLs from Streams config section, filtered by protocol.

    Falls back to legacy RTMP section (audio_streams / video_streams) if Streams is empty.
    """
    streams = config.get("Streams", {})
    if isinstance(streams, dict) and streams:
        urls = []
        for entry in streams.values():
            if isinstance(entry, dict):
                target = entry.get("target", "")
                if target.startswith(f"{protocol}://"):
                    urls.append(target)
        if urls:
            return urls

    rtmp = config.get("RTMP", {})
    if isinstance(rtmp, dict):
        for key in ("audio_streams", "video_streams"):
            val = rtmp.get(key)
            if isinstance(val, list):
                return [u for u in val if isinstance(u, str) and u.startswith(f"{protocol}://")]
            if isinstance(val, str) and "," in val:
                return [u.strip() for u in val.split(",") if u.strip().startswith(f"{protocol}://")]
    return []
