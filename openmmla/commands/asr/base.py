import argparse
import functools


def get_parser():
    parser = argparse.ArgumentParser(
        prog="mmla asr-base",
        description="Run ASR base of real-time audio analyzer",
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=80, width=150)
    )
    from openmmla.utils.args import add_arguments
    add_arg = functools.partial(add_arguments, argparser=parser)
    add_arg('project_dir', str, None,
            'path to the project directory; if not set, defaults to the current working directory', shortname='-p')
    add_arg('config_path', str, None, 'path to the configuration file', shortname='-c', required=True)
    add_arg('mode', str, 'capture', 'operating mode (capture/analyze/live)', choices=['capture', 'analyze', 'live'], shortname='-m')
    add_arg('store', bool, True, 'whether to store audio', shortname='-s')
    add_arg('vad', bool, True, 'whether to use VAD', shortname='-vad')
    add_arg('nr', bool, True, 'whether to use noise reduction', shortname='-nr')
    add_arg('tr', bool, True, 'whether to transcribe speech to text', shortname='-tr')
    add_arg('sp', bool, False, 'whether to do speech separation', shortname='-sp')
    add_arg('hsr', bool, True, 'whether to apply Half-Scaled Recognition at speaker boundaries', shortname='-hsr')
    add_arg('session_id', str, None,
            'session id to use; if set, start recognizing at once without the menu and exit when the run '
            'ends (STOP); if not set, choose/create one interactively', shortname='-sid')
    add_arg('base', str, None,
            "base name from config 'Bases' (pulls base_type/id/device from that entry); "
            "if not set, the only entry when started with -sid, else choose from the Bases list interactively",
            shortname='-b')
    add_arg('speakers', str, None,
            "comma-separated speaker profiles to recognize (see mmla asr-speakers --list); if not set, every "
            "registered one", shortname='-spk')
    add_arg('participant', str, None,
            "whom this base's speech is attributed to: a participant's tag id (a microphone they wear), "
            "'group' (the session's group) or 'speakers' (speaker verification among -spk); if not set, the "
            "config decides (asr_scope, the Bases entry's participant, the wearer the session's Collection Start "
            "picked for its stream)", shortname='-pt')
    add_arg('language', str, None,
            "language to transcribe this base's speech in ('en', 'da', 'zh-CN'): sent with every request and "
            "taken for it alone, whatever the speech transcriber is configured for; if not set, that "
            "configured language", shortname='-lang')
    add_arg('diarize', bool, False,
            "whether every chunk is sent for its anonymous speaker turns (pyannote diarization on the speech "
            "transcriber, local WhisperX models only): the transcript record then carries who-of-how-many spoke "
            "when as SPEAKER_00, SPEAKER_01 ..., without names or profiles, each linked to a voice of the "
            "session (1, 2, 3 ...) by its speaker embedding", shortname='-dia')
    return parser


def main():
    """Main function."""
    parser = get_parser()
    args = parser.parse_args()
    
    # Only import when actually running the logic
    from openmmla.bases.asr import start_asr_base
    from openmmla.utils.args import print_arguments

    print_arguments(args)
    
    # Call the centralized start_asr_base function with restart capability
    start_asr_base(
        project_dir=args.project_dir,
        config_path=args.config_path,
        mode=args.mode,
        store=args.store,
        vad=args.vad,
        nr=args.nr,
        tr=args.tr,
        sp=args.sp,
        hsr=args.hsr,
        session_id=args.session_id,
        base=args.base,
        speakers=args.speakers,
        participant=args.participant,
        language=args.language,
        diarize=args.diarize
    )


if __name__ == "__main__":
    main()
