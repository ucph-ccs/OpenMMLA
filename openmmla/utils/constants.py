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

# grouped by pipeline for management operations
EVENT_TYPES_ASR = [EVENT_TYPE_ASR_TRANSCRIPTION, EVENT_TYPE_ASR_RECOGNITION]
EVENT_TYPES_IPS = [EVENT_TYPE_IPS_TRANSLATION, EVENT_TYPE_IPS_ROTATION, EVENT_TYPE_IPS_RELATION]
EVENT_TYPES_VFA = [EVENT_TYPE_VFA_ACTION]
EVENT_TYPES_ALL = EVENT_TYPES_ASR + EVENT_TYPES_IPS + EVENT_TYPES_VFA

# mongodb defaults
MONGODB_DEFAULT_DB = "openmmla"
