"""shared infrastructure sections that appear across multiple pipeline configs.

When users fill these once in the "Global Defaults" panel, the values propagate
to every pipeline config that contains the matching section name.
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
