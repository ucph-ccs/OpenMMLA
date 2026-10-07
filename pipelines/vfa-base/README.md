# VFA base station bundle

Runtime files for the VFA (video frame analysis with VLMs/LLMs) pipeline: `config_template.yml` is the reference configuration, `config.yml` (gitignored) is the live one the TUI edits and syncs to base stations. `examples/analyze_video_frame.py` sends still frames to a running VFA server for a smoke test, and `coding-interface/` holds the human coding page and its action template ([docs/pipelines/vfa/coding_interface.md](../../docs/pipelines/vfa/coding_interface.md)).

The pipeline guide starts at [docs/pipelines/vfa/index.md](../../docs/pipelines/vfa/index.md): the launch steps are in [run.md](../../docs/pipelines/vfa/run.md), the backend choice and prompt templates in [action-labels.md](../../docs/pipelines/vfa/action-labels.md), and every config key in [configuration.md](../../docs/pipelines/vfa/configuration.md).
