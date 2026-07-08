"""system service sections shared by OpenMMLA pipeline configs.

These values describe where the common deployment services live. The TUI uses
them to prefill and synchronize matching sections across pipeline configs.
"""

SHARED_SECTIONS = {
    "InfluxDB": {
        "label": "InfluxDB",
        "fields": {
            "url": {
                "description": "InfluxDB server URL (e.g. http://localhost:8086)",
                "field_type": "url",
                "default": "http://localhost:8086",
            },
            "token": {
                "description": "InfluxDB authentication token",
                "field_type": "str",
                "default": "",
            },
            "org": {
                "description": "InfluxDB organization name",
                "field_type": "str",
                "default": "",
            },
            "bucket": {
                "description": "InfluxDB bucket name (default mmla-data)",
                "field_type": "str",
                "default": "mmla-data",
            },
        },
    },
    "MongoDB": {
        "label": "MongoDB",
        "fields": {
            "url": {
                "description": "MongoDB connection URL (e.g. mongodb://localhost:27017)",
                "field_type": "url",
                "default": "mongodb://localhost:27017",
            },
            "db": {
                "description": "MongoDB database name (default openmmla)",
                "field_type": "str",
                "default": "openmmla",
            },
        },
    },
    "MQTT": {
        "label": "MQTT",
        "fields": {
            "host": {
                "description": "MQTT broker hostname (e.g. localhost)",
                "field_type": "str",
                "default": "localhost",
            },
            "port": {
                "description": "MQTT broker port (default 1883)",
                "field_type": "int",
                "default": 1883,
            },
        },
    },
    "Redis": {
        "label": "Redis",
        "fields": {
            "host": {
                "description": "Redis server hostname (e.g. localhost)",
                "field_type": "str",
                "default": "localhost",
            },
            "port": {
                "description": "Redis server port (default 6379)",
                "field_type": "int",
                "default": 6379,
            },
            "db": {
                "description": "Redis database number (default 0)",
                "field_type": "int",
                "default": 0,
            },
        },
    },
    "Sudo": {
        "label": "Sudo (local admin)",
        "fields": {
            "password": {
                "description": "Local sudo password, auto-filled when privileged Start/Stop commands prompt for it (stored encrypted; leave empty to type manually)",
                "field_type": "str",
                "default": "",
            },
        },
    },
    "Gateway": {
        "label": "Gateway (Nginx)",
        "fields": {
            "host": {
                "description": "Nginx gateway host that bases connect to (e.g. localhost)",
                "field_type": "str",
                "default": "localhost",
            },
            "http_port": {
                "description": "Nginx HTTP reverse-proxy port (default 8080)",
                "field_type": "int",
                "default": 8080,
            },
            "rtmp_port": {
                "description": "Nginx RTMP port for video/audio streams (default 1935)",
                "field_type": "int",
                "default": 1935,
            },
            "scheme": {
                "description": "URL scheme for HTTP services (http or https)",
                "field_type": "str",
                "default": "http",
            },
        },
    },
}


SHARED_SECTION_NAMES = set(SHARED_SECTIONS.keys())


def get_shared_defaults():
    """return a flat dict of shared section defaults: { 'InfluxDB.url': 'http://...', ... }"""
    defaults = {}
    for section, info in SHARED_SECTIONS.items():
        for key, fdef in info["fields"].items():
            defaults[f"{section}.{key}"] = fdef["default"]
    return defaults


def apply_shared_values(pipeline_fields, shared_values):
    """for each pipeline field whose section is shared, override its default with
    the user-provided shared value if one exists."""
    for f in pipeline_fields:
        if f.path in shared_values and shared_values[f.path] is not None:
            f.default = shared_values[f.path]
