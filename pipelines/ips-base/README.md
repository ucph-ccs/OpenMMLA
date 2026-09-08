# IPS base station bundle

Runtime files for the IPS (indoor positioning with AprilTags) pipeline: `config_template.yml` is the reference configuration, `config.yml` (gitignored) is the live one the TUI edits and syncs to base stations. `apriltag/` holds printable tags, `camera_calib/` and `camera_sync/` receive the calibration images and transform matrices, and `docs/clock.html` is a browser clock you can film to check timing.

The pipeline guide, including camera calibration, multi-camera sync and the launch steps, is in [docs/pipelines/ips.md](../../docs/pipelines/ips.md).
