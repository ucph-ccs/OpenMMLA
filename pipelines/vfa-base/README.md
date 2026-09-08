# VFA base station bundle

Runtime files for the VFA (video frame analysis with VLMs/LLMs) pipeline: `config_template.yml` is the reference configuration, `config.yml` (gitignored) is the live one the TUI edits and syncs to base stations. `examples/analyze_video_frame.py` sends still frames to a running VFA server for a smoke test.

The pipeline guide, including backend choice, prompt templates and the launch steps, is in [docs/pipelines/vfa.md](../../docs/pipelines/vfa.md).
