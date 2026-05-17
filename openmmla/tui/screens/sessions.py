from __future__ import annotations

import os
from datetime import datetime, timezone

from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Vertical, Horizontal
from textual.widget import Widget
from textual.widgets import Static, DataTable, RichLog, Button


def _find_config_path() -> str | None:
    """find the first pipeline config.yml that contains MongoDB and InfluxDB sections."""
    from openmmla.tui.schema.loader import discover_pipelines, load_existing_config
    for pipeline in discover_pipelines():
        data = load_existing_config(pipeline.config_path)
        if data.get("MongoDB") and data.get("InfluxDB"):
            return pipeline.config_path
    return None


class SessionsPanel(Widget):

    DEFAULT_CSS = """
    SessionsPanel {
        width: 1fr;
        height: 1fr;
    }
    #sessions-summary {
        height: 3;
        padding: 0 2;
        background: $panel;
        content-align: center middle;
        text-style: bold;
    }
    #sessions-table {
        height: 1fr;
    }
    #sessions-actions {
        height: 3;
        padding: 0 1;
    }
    #sessions-actions Button {
        margin: 0 1;
    }
    #sessions-log {
        height: 14;
        border-top: solid $primary;
        padding: 0 1;
    }
    """

    def __init__(self) -> None:
        super().__init__()
        self._config_path: str | None = None
        self._mongo_client = None
        self._influx_client = None
        self._sessions: list[dict] = []
        self._selected_session_id: str | None = None

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Discovering database configuration...", id="sessions-summary")
            yield DataTable(id="sessions-table")
            with Horizontal(id="sessions-actions"):
                yield Button("Refresh", variant="primary", id="btn-ses-refresh")
                yield Button("Export Logs", variant="success", id="btn-ses-export-logs")
                yield Button("Export Visualizations", variant="success", id="btn-ses-export-vis")
                yield Button("Export All", variant="warning", id="btn-ses-export-all")
            yield RichLog(id="sessions-log", highlight=True, markup=True)

    def on_mount(self) -> None:
        table = self.query_one("#sessions-table", DataTable)
        table.add_columns("Session ID", "Experiment", "Group", "Status", "Started")
        table.cursor_type = "row"
        self.run_worker(self._async_init(), exclusive=True)

    async def _async_init(self) -> None:
        """initialize db clients and load sessions in a background worker."""
        import asyncio
        loop = asyncio.get_event_loop()
        config_path = await loop.run_in_executor(None, _find_config_path)

        if not config_path:
            self._update_summary("No pipeline config with MongoDB/InfluxDB found. Configure a pipeline first.")
            return

        self._config_path = config_path

        try:
            from openmmla.utils.client import MongoDBClientWrapper, InfluxDBClientWrapper
            self._mongo_client = await loop.run_in_executor(
                None, MongoDBClientWrapper, config_path,
            )
            self._influx_client = await loop.run_in_executor(
                None, InfluxDBClientWrapper, config_path,
            )
        except Exception as e:
            self._update_summary(f"Database connection failed: {e}")
            return

        self._refresh_sessions()

    # ---- UI helpers ----

    def _update_summary(self, text: str) -> None:
        try:
            summary = self.query_one("#sessions-summary", Static)
            summary.update(Text(str(text)))
        except Exception:
            pass

    def _log(self, msg: str) -> None:
        try:
            log = self.query_one("#sessions-log", RichLog)
            log.write(msg)
        except Exception:
            pass

    # ---- session listing ----

    def _refresh_sessions(self) -> None:
        if not self._mongo_client:
            self._update_summary("Not connected to MongoDB.")
            return

        self._sessions = self._mongo_client.get_all_sessions()
        table = self.query_one("#sessions-table", DataTable)
        table.clear()

        for ses in self._sessions:
            sid = ses.get("session_id", "")
            exp = ses.get("experiment_id", "")
            grp = ses.get("group_id", "")
            status = ses.get("status", "unknown")
            start = ses.get("start_time")
            start_str = start.strftime("%Y-%m-%d %H:%M UTC") if isinstance(start, datetime) else str(start or "-")
            table.add_row(sid, exp, grp, status, start_str)

        self._update_summary(f"Sessions: {len(self._sessions)} found")

    # ---- event handlers ----

    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        table = self.query_one("#sessions-table", DataTable)
        try:
            row_data = table.get_row(event.row_key)
            self._selected_session_id = str(row_data[0])
        except Exception:
            self._selected_session_id = None

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-ses-refresh":
            self._refresh_sessions()
            return

        if not self._selected_session_id:
            self._log("[yellow]Select a session row first.[/yellow]")
            return

        session_id = self._selected_session_id

        if bid == "btn-ses-export-logs":
            self.run_worker(self._run_export(session_id, logs=True, vis=False), exclusive=True)
        elif bid == "btn-ses-export-vis":
            self.run_worker(self._run_export(session_id, logs=True, vis=True), exclusive=True)
        elif bid == "btn-ses-export-all":
            self.run_worker(self._run_export(session_id, logs=True, vis=True), exclusive=True)

    # ---- export logic ----

    async def _run_export(self, session_id: str, *, logs: bool, vis: bool) -> None:
        import asyncio
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._do_export, session_id, logs, vis)

    def _do_export(self, session_id: str, logs: bool, vis: bool) -> None:
        from openmmla.utils.constants import (
            EVENT_TYPE_ASR_RECOGNITION, EVENT_TYPE_ASR_TRANSCRIPTION,
            EVENT_TYPE_VFA_ACTION,
            EVENT_TYPE_IPS_TRANSLATION, EVENT_TYPE_IPS_ROTATION, EVENT_TYPE_IPS_RELATION,
        )
        from openmmla.utils.querys import fetch_and_process_data, save_to_json_file

        cwd = os.getcwd()
        logs_dir = os.path.join(cwd, "logs")
        log_dir = os.path.join(logs_dir, session_id)
        vis_dir = os.path.join(cwd, "visualizations", session_id, "post-time")
        vis_root = os.path.join(cwd, "visualizations")

        self._log(f"[bold]Exporting session: {session_id}[/bold]")
        self._log(f"Output directory: {cwd}")

        # ---- export logs ----
        if logs:
            os.makedirs(log_dir, exist_ok=True)

            event_types = {
                "ASR Recognition": (EVENT_TYPE_ASR_RECOGNITION, "speaker_recognition"),
                "ASR Transcription": (EVENT_TYPE_ASR_TRANSCRIPTION, "speaker_transcription"),
                "VFA Action": (EVENT_TYPE_VFA_ACTION, "action_recognition"),
                "IPS Translation": (EVENT_TYPE_IPS_TRANSLATION, "badge_translation"),
                "IPS Rotation": (EVENT_TYPE_IPS_ROTATION, "badge_rotation"),
                "IPS Relation": (EVENT_TYPE_IPS_RELATION, "badge_relation"),
            }

            exported_files: dict[str, str] = {}
            for label, (evt, suffix) in event_types.items():
                try:
                    data = fetch_and_process_data(session_id, evt, self._influx_client)
                    if data:
                        path = save_to_json_file(session_id, data, suffix, log_dir)
                        exported_files[label] = path
                        self._log(f"  [green]✓[/green] {label}: {len(data)} records -> {os.path.basename(path)}")
                    else:
                        self._log(f"  [dim]- {label}: no data[/dim]")
                except Exception as e:
                    self._log(f"  [red]✗ {label}: {e}[/red]")

            # convert transcription to txt
            if "ASR Transcription" in exported_files:
                try:
                    from openmmla.analytics.asr.transcription import convert_transcription_json_to_txt
                    convert_transcription_json_to_txt(exported_files["ASR Transcription"])
                    self._log("  [green]✓[/green] ASR Transcription -> .txt")
                except Exception as e:
                    self._log(f"  [red]✗ Transcription txt conversion: {e}[/red]")

        # ---- export visualizations ----
        if vis:
            os.makedirs(vis_dir, exist_ok=True)

            # ASR visualizations
            recognition_path = os.path.join(log_dir, f"{session_id}_speaker_recognition.json")
            if os.path.isfile(recognition_path):
                try:
                    from openmmla.analytics.asr.analyze import (
                        plot_speaker_diarization_interactive,
                        plot_speaking_interaction_network,
                        asr_across_sessions_analysis,
                    )
                    plot_speaker_diarization_interactive(recognition_path, vis_dir)
                    self._log("  [green]✓[/green] ASR: speaker diarization chart")
                    plot_speaking_interaction_network(recognition_path, vis_dir)
                    self._log("  [green]✓[/green] ASR: speaking interaction network")
                    asr_across_sessions_analysis(logs_dir, vis_root)
                    self._log("  [green]✓[/green] ASR: cross-session analysis")
                except Exception as e:
                    self._log(f"  [red]✗ ASR visualizations: {e}[/red]")
            else:
                self._log("  [dim]- ASR visualizations: no recognition log file[/dim]")

            # IPS visualizations
            translation_path = os.path.join(log_dir, f"{session_id}_badge_translation.json")
            relation_path = os.path.join(log_dir, f"{session_id}_badge_relation.json")
            if os.path.isfile(translation_path):
                try:
                    from openmmla.analytics.ips.analyze import (
                        plot_badge_locations_and_trajectories,
                        plot_2d_heatmap,
                        plot_physical_interaction_network,
                        ips_across_sessions_analysis,
                    )
                    plot_badge_locations_and_trajectories(translation_path, vis_dir)
                    self._log("  [green]✓[/green] IPS: badge locations and trajectories")
                    plot_2d_heatmap(translation_path, vis_dir)
                    self._log("  [green]✓[/green] IPS: 2D heatmap")
                    if os.path.isfile(relation_path):
                        plot_physical_interaction_network(relation_path, vis_dir)
                        self._log("  [green]✓[/green] IPS: physical interaction network")
                    ips_across_sessions_analysis(logs_dir, vis_root)
                    self._log("  [green]✓[/green] IPS: cross-session analysis")
                except Exception as e:
                    self._log(f"  [red]✗ IPS visualizations: {e}[/red]")
            else:
                self._log("  [dim]- IPS visualizations: no translation log file[/dim]")

            # VFA has no visualizations currently
            self._log("  [dim]- VFA visualizations: not available[/dim]")

        self._log(f"[bold green]Export complete for {session_id}[/bold green]")
