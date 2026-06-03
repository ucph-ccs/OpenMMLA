import argparse
import os


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-gd-export",
        description="Export OpenMMLA-GD windowed group dynamics datasets from session artifacts.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150),
    )
    parser.add_argument("--session-id", required=True, help="session id to export")
    parser.add_argument("--config-path", default=None, help="configuration path, required for source=influx or source=both")
    parser.add_argument("--project-dir", default=None, help="project directory used to resolve relative paths")
    parser.add_argument("--artifacts-root", default="artifacts", help="artifacts root, relative to project dir unless absolute")
    parser.add_argument("--output-dir", default=None, help="output directory; defaults to artifacts/<session_id>/analysis/group_dynamics")
    parser.add_argument("--source", choices=["artifacts", "influx", "both"], default="artifacts", help="measurement event source")
    parser.add_argument("--window-size", type=float, default=30.0, help="window size in seconds")
    parser.add_argument("--step-size", type=float, default=15.0, help="window step in seconds")
    parser.add_argument("--participants", default="", help="comma-separated participant or tag ids; inferred when omitted")
    parser.add_argument("--group-id", default=None, help="group id override")
    parser.add_argument("--audio-scope", choices=["participant", "group", "none"], default="group", help="audio capture scope")
    return parser


def run_group_dynamics_export(args):
    from openmmla.analytics.group_dynamics import export_group_dynamics_dataset

    project_dir = args.project_dir or os.getcwd()
    if not os.path.isabs(project_dir):
        project_dir = os.path.abspath(project_dir)

    config_path = args.config_path
    if config_path and not os.path.isabs(config_path):
        config_path = os.path.join(project_dir, config_path)

    participants = [
        item.strip()
        for item in str(args.participants or "").split(",")
        if item.strip()
    ]

    result = export_group_dynamics_dataset(
        project_dir=project_dir,
        session_id=args.session_id,
        config_path=config_path,
        artifacts_root=args.artifacts_root,
        output_dir=args.output_dir,
        source=args.source,
        window_size=args.window_size,
        step_size=args.step_size,
        participants=participants,
        group_id=args.group_id,
        audio_scope=args.audio_scope,
    )
    print(f"OpenMMLA-GD dataset exported: {result.window_count} windows")
    print(f"windows: {result.windows_path}")
    print(f"manifest: {result.manifest_path}")
    return result


def main():
    parser = get_parser()
    args = parser.parse_args()
    run_group_dynamics_export(args)


if __name__ == "__main__":
    main()
