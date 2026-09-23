import argparse
import functools
import os


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla ses-fuse",
        description="Build the fusion table of a session: one row per time window with the speech, space, body, "
                    "gaze and action features side by side, from InfluxDB or from an exported measurements folder.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None,
            'path to a configuration file with an InfluxDB section (read to fetch the events); not needed with '
            '--measurements', shortname='-c')
    add_arg('session_id', str, None, 'the session to build the table of', shortname='-sid')
    add_arg('measurements', str, None,
            'a Sessions -> Export Measurements folder (artifacts/<session>/measurements) to read the events from '
            'instead of InfluxDB', shortname='-md')
    add_arg('window', float, 10.0, 'window length in seconds', shortname='-w')
    add_arg('step', float, 10.0, 'step between windows in seconds (equal to the window for no overlap)', shortname='-st')
    add_arg('participants', str, None,
            "comma-separated tag ids to build columns for, and the session's pupils; if not set, the tags the events "
            "hold, and the pupils the session's manifest declares (else the tags up to 12)", shortname='-tags')
    add_arg('out', str, None,
            'where to write the table (.csv, else JSON lines); if not set, '
            'artifacts/<session>/analysis/features/<session>_window_features.csv', shortname='-o')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    from openmmla.utils.args import print_arguments
    print_arguments(args)

    from openmmla.analytics.fusion import window_features as fusion
    from openmmla.analytics.fusion.window_features import (
        export_files, load_events_from_export, load_events_from_influx, session_of_export, write_table,
    )
    if args.window <= 0 or args.step <= 0:
        parser.error("-w/--window and -st/--step must be greater than 0")
    project_dir = args.project_dir or os.getcwd()
    session_id = args.session_id
    inputs: list[str] = []
    if args.measurements:
        # with -sid only that session's files are read; without it every export file in the
        # folder is, and the table is named after the session the folder holds
        events = load_events_from_export(args.measurements, session_id)
        inputs = [path for paths in export_files(args.measurements, session_id).values() for path in paths]
        if not session_id:
            session_id = session_of_export(args.measurements)
            if not session_id:
                parser.error(f"could not tell the session from {args.measurements}: give -sid <session>")
    else:
        if not args.config_path or not session_id:
            parser.error("give -c <config with an InfluxDB section> and -sid <session>, or --measurements <folder>")
        from openmmla.utils.client import InfluxDBClientWrapper
        events = load_events_from_influx(session_id, InfluxDBClientWrapper(args.config_path))

    participants = [t.strip() for t in args.participants.split(',') if t.strip()] if args.participants else None
    # the session's pupils, the in-group set whose faces and hands are a partner's: the -tags, else the
    # pupils its manifest declares, else (None) the tags up to the IPS trust bound
    from openmmla.analytics.interaction.layout import declared_pupils
    from openmmla.utils.artifact_paths import session_artifact_dir
    session_dir = session_artifact_dir(project_dir, session_id)
    if participants is not None:
        pupils, pupils_source = list(participants), 'tags'
    else:
        try:
            pupils = declared_pupils(session_dir)
        except ValueError as e:
            parser.error(str(e))
        pupils_source = 'manifest' if pupils is not None else 'trust bound'
    rows = fusion.window_features(events, window=args.window, step=args.step, participants=participants, pupils=pupils)
    if not rows:
        print(f"No events found for session {session_id}: nothing to build a table from.")
        return
    used_pupils = pupils if pupils is not None else fusion.default_pupils(fusion.participants_of(events))
    print(f"Pupils ({pupils_source}): {', '.join(used_pupils) or 'none'}")

    out = args.out
    if not out:
        out = os.path.join(os.fspath(session_dir), 'analysis', 'features', f'{session_id}_window_features.csv')
    path = write_table(rows, out)
    counts = {event_type: len(records) for event_type, records in events.items() if records}
    print(f"{len(rows)} windows of {args.window:g} s ({len(rows[0])} columns) from {counts} -> {path}")

    # what the table was made from and with, next to it
    try:
        from openmmla.services.vfa.work_area import WORK_AREA_HIGH, WORK_AREA_LOW, WORK_AREA_MIN_HANDS
        from openmmla.utils.session_provenance import analysis_record, write_analysis_record
        record = analysis_record(session_id, inputs=inputs, outputs=[path], steps=['fusion.window_features'],
                                 parameters={'window': args.window, 'step': args.step, 'participants': participants,
                                             'pupils': used_pupils, 'pupils_source': pupils_source,
                                             'work_area': {'low': WORK_AREA_LOW, 'high': WORK_AREA_HIGH,
                                                           'min_hands': WORK_AREA_MIN_HANDS,
                                                           'pad': 'median hand radius + max(W,H)/64'},
                                             'seat_partners': 'an untagged gaze target at the seat of a pupil '
                                                              'missing from the frame is that pupil',
                                             'joint_baseline': {'lags': list(fusion.JOINT_BASELINE_LAGS),
                                                                'slack': fusion.JOINT_BASELINE_SLACK,
                                                                'min': fusion.JOINT_BASELINE_MIN},
                                             'events': counts, 'source': args.measurements or 'influxdb'},
                                 root=project_dir, project_dir=project_dir)
        write_analysis_record(record, os.path.join(os.path.dirname(path), 'fusion'))
    except Exception as e:  # the table stands without its record
        print(f"Could not write the analysis record next to the table: {e}")


if __name__ == "__main__":
    main()
