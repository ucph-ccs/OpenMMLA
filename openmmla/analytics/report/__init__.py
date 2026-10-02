"""The dashboard's report of a session, computed from the raw events its pipelines wrote to
InfluxDB (and, when MongoDB answers, from the session document).

`common` reads the events and holds the shared conventions (session offsets, tag order, voice
keys); `sessions` lists the sessions and describes one; `live` slims records for the live
stream; `exports` writes the downloads; `speech`, `space` and `video` build the parts of the
analysis page. Every public function returns plain JSON-safe structures, so the server can cache
and send them as they are. The modules import nothing from each other at package import, so a
caller pays only for what it uses."""

# the version of what the parts hold: a cached part written by another version is recomputed, so
# raise it whenever a change alters what a part computes (not for a change that only moves code)
REPORT_VERSION = 5
