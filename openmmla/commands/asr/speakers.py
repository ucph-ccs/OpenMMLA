"""mmla asr-speakers: the speaker profiles the ASR bases of this host recognize.

It asks nothing, so it runs without a terminal: the Launcher runs it on a
card's host (over SSH for another one) and reads the last line it prints,
`@speakers <json>`, for the profiles there. --register says what it does as it
goes, and exits with 1 when nothing was registered."""

import argparse
import functools
import json
import os
import sys

# the line the Launcher reads the answer from
MARKER = "@speakers "


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla asr-speakers",
        description="List, register and delete the speaker profiles the ASR bases of this host recognize.",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory (as for mmla asr-base); if not set, the current working directory',
            shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file; needed to register', shortname='-c')
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument('--list', action='store_true', help='list the registered speaker profiles')
    action.add_argument('--delete', nargs='+', metavar='name', help='delete the profiles of these speakers')
    action.add_argument('--register', metavar='name',
                        help="register this speaker, or add to their profile: from --files, else recorded "
                             "from the source of base -b")
    parser.add_argument('--files', nargs='+', metavar='file',
                        help='reference audio files on this host to register from, instead of recording')
    add_arg('base', str, None,
            "id of the config 'Bases' entry whose source records the speaker and whose base type's settings "
            "prepare the audio; needed to register", shortname='-b')
    add_arg('duration', float, None, "seconds to record; if not set, the base type's register_duration",
            shortname='-t')
    add_arg('vad', bool, True, 'whether to keep only the speech of the audio (VAD)', shortname='-vad')
    add_arg('nr', bool, True, 'whether to reduce the noise of the audio', shortname='-nr')
    add_arg('store', bool, True, 'whether to keep the audio of the registration in the profile', shortname='-s')
    return parser


def _answer(directory: str, **fields) -> None:
    from openmmla.utils.artifact_paths import short_hostname
    print(MARKER + json.dumps({"host": short_hostname(), "dir": directory, **fields}, ensure_ascii=False),
          flush=True)


def main():
    """Main function."""
    args = get_parser().parse_args()
    # the Launcher reads this through a pipe: every line as it comes
    sys.stdout.reconfigure(line_buffering=True)

    from openmmla.bases.asr.speaker_profiles import delete_speakers, list_speakers, profiles_dir

    directory = profiles_dir(args.project_dir)
    if args.list:
        _answer(directory, speakers=list_speakers(directory))
        return
    if args.delete:
        deleted, missing = delete_speakers(directory, args.delete)
        for name in deleted:
            print(f"Deleted the speaker profile of '{name}'.")
        for name in missing:
            print(f"There is no speaker profile of '{name}' here.")
        _answer(directory, deleted=deleted, missing=missing, speakers=list_speakers(directory))
        return

    name = args.register
    try:
        if not args.config_path or not args.base:
            raise ValueError("Registering a speaker needs the config (-c) and the base whose settings it takes (-b).")
        from openmmla.bases.asr.asr_base import ASRBase
        base = ASRBase(project_dir=args.project_dir, config_path=args.config_path, mode='live',
                       vad=args.vad, nr=args.nr, store=args.store, base=args.base,
                       registration='files' if args.files else 'stream')
        if args.files:
            used, embeddings = base.register_speaker_from_files(name, [os.path.abspath(path) for path in args.files])
            print(f"Registered '{name}' from {used} file(s): the profile holds {embeddings} embedding(s).")
        else:
            embeddings = base.register_speaker_from_stream(name, args.duration)
            print(f"Registered '{name}': the profile holds {embeddings} embedding(s).")
    except Exception as e:
        print(f"Could not register '{name}': {e}")
        _answer(directory, error=str(e), speakers=list_speakers(directory))
        sys.exit(1)
    _answer(directory, registered=name, speakers=list_speakers(directory))


if __name__ == "__main__":
    main()
