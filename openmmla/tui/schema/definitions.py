"""system service sections shared by OpenMMLA pipeline configs.

These values describe where the common deployment services live. The TUI uses
them to prefill and synchronize matching sections across pipeline configs.

The machine of every address defaults to the placeholder of
config/system_services_template.yml, <uber-server>: no default can know which
machine runs the services, and localhost would read as filled in while it
means the very machine that reads it. Until a form is filled, its address is
not set (system_services.usable_system_service_value).
"""

# what an address of System Settings holds until it is filled in
UBER_SERVER_PLACEHOLDER = "<uber-server>"

SHARED_SECTIONS = {
    "InfluxDB": {
        "label": "InfluxDB",
        "fields": {
            "url": {
                "description": "URL of the InfluxDB server, by the name the machines of the deployment reach it "
                               "(localhost only when everything runs on this machine)",
                "field_type": "url",
                "default": f"http://{UBER_SERVER_PLACEHOLDER}:8086",
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
                "description": "MongoDB connection URL, by the name the machines of the deployment reach the "
                               "server (localhost only when everything runs on this machine)",
                "field_type": "url",
                "default": f"mongodb://{UBER_SERVER_PLACEHOLDER}:27017",
            },
            "db": {
                "description": "MongoDB database name (default openmmla)",
                "field_type": "str",
                "default": "openmmla",
            },
        },
    },
    "MQTT": {
        "label": "MQTT (Mosquitto)",
        "fields": {
            "host": {
                "description": "Host the MQTT broker runs on, by the name the machines of the deployment reach "
                               "it (localhost only when everything runs on this machine)",
                "field_type": "str",
                "default": UBER_SERVER_PLACEHOLDER,
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
                "description": "Host the Redis server runs on, by the name the machines of the deployment reach "
                               "it (localhost only when everything runs on this machine)",
                "field_type": "str",
                "default": UBER_SERVER_PLACEHOLDER,
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
    "Dashboard": {
        "label": "Dashboard (Flask)",
        "fields": {
            "host": {
                "description": "Host the dashboard backend runs on, as reached from this machine (localhost when "
                               "it runs on this one)",
                "field_type": "str",
                "default": UBER_SERVER_PLACEHOLDER,
            },
            "port": {
                "description": "Dashboard backend (gunicorn) port; `make flask` binds to it and to the next port, "
                               "the media port the Live page loads recorded video from (default 5050)",
                "field_type": "int",
                "default": 5050,
            },
        },
    },
    "Gateway": {
        "label": "Gateway (Nginx)",
        "fields": {
            "host": {
                "description": "Host of the Nginx load balancer that the bases reach the AI services through, by "
                               "the name the bases reach it (localhost only when everything runs on this machine)",
                "field_type": "str",
                "default": UBER_SERVER_PLACEHOLDER,
            },
            "http_port": {
                "description": "Nginx HTTP reverse-proxy port (default 8080)",
                "field_type": "int",
                "default": 8080,
            },
            "scheme": {
                "description": "URL scheme for HTTP services (http or https)",
                "field_type": "str",
                "default": "http",
            },
        },
    },
    # a section of its own: the stream server need not share a machine with the
    # load balancer. Only the console reads it (where the MediaMTX card runs,
    # what it probes, and what completes a stream written as a bare path);
    # streams and bases use the full URLs of their Streams entries, so it is
    # not copied into the pipeline configs
    "StreamServer": {
        "label": "Stream Server (MediaMTX)",
        "fields": {
            "host": {
                "description": "Host MediaMTX runs on, by the name cameras and bases on other machines reach it (localhost only when everything runs here). A stream written as a path (ips/cam-1) is completed with it",
                "field_type": "str",
                "default": UBER_SERVER_PLACEHOLDER,
            },
            "rtmp_port": {
                "description": "MediaMTX RTMP port that cameras and microphones publish to (default 1935)",
                "field_type": "int",
                "default": 1935,
            },
            "rtsp_port": {
                "description": "MediaMTX RTSP port that bases pull streams from (default 8554)",
                "field_type": "int",
                "default": 8554,
            },
            "api_port": {
                "description": "MediaMTX control API port, asked what was recorded and for how long it is kept, and told what to delete (default 9997)",
                "field_type": "int",
                "default": 9997,
            },
            "playback_port": {
                "description": "MediaMTX playback port, which hands out the recording of a time range: Sessions → Export (default 9996)",
                "field_type": "int",
                "default": 9996,
            },
            "webrtc_port": {
                "description": "MediaMTX WebRTC port browsers watch live video on (the dashboard's camera tiles)",
                "field_type": "int",
                "default": 8889,
            },
        },
    },
}

# sections only a console reads: no pipeline config carries them and no
# service reads them at startup. They reach another machine through its own
# config/system_services.yml, which a console there reads; the stream
# server's address also travels as the stream URLs it completed
CONSOLE_ONLY_SECTIONS = frozenset({"Sudo", "StreamServer"})

# never leaves this machine
PRIVATE_SECTIONS = frozenset({"Sudo"})


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
