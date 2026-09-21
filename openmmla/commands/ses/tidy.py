"""mmla ses-tidy: a session under artifacts/ renamed, its hosts relabelled, its legacy folder
reduced to what is raw.

The collection layout names a session <experiment>_<group>_<start> and every recording after
its host: audio_<host>_<channel>_<start>.wav, video_<host>_<device>_<start>.<ext>. When the
experiment or the group was wrong, or a host label should be the device's kind (vimo-0 rather
than base-vimo-0), the files, the folders and every path in the manifests move together.
--prune-legacy keeps a session's speaker profiles (moved next to their base's recordings as
collection/<host>/profiles/) and its meta.txt, and deletes the rest of legacy/: the frames, the
logs and the measurements an earlier run produced, which a replay produces again.
"""
import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

SESSION_ID_RE = re.compile(r'^(?P<experiment>.+?)_(?P<group>group_[^_]+)_(?P<start>\d{6}T\d{4}Z)$')
KEEP_IN_LEGACY_ROOT = ('meta.txt',)


def split_session_id(session_id: str) -> dict[str, str]:
    match = SESSION_ID_RE.match(session_id)
    if not match:
        raise ValueError(f"{session_id} is not <experiment>_<group>_<YYMMDDTHHMMZ>")
    return match.groupdict()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}


def _write(path: Path, data: dict[str, Any]) -> None:
    from openmmla.collection.recording import _dump_yaml
    path.write_text(json.dumps(data, indent=2) + "\n", encoding='utf-8')
    path.with_suffix('.yml').write_text("\n".join(_dump_yaml(data)) + "\n", encoding='utf-8')


def _replace_paths(value: Any, old: str, new: str) -> Any:
    """every string in a manifest that starts with the old path, moved to the new one"""
    if isinstance(value, str):
        return new + value[len(old):] if value.startswith(old) else value
    if isinstance(value, list):
        return [_replace_paths(v, old, new) for v in value]
    if isinstance(value, dict):
        return {k: _replace_paths(v, old, new) for k, v in value.items()}
    return value


def _rewrite_manifests(session_dir: Path, transform) -> None:
    """every manifest and report of the session passed through `transform(data) -> data`"""
    for path in [session_dir / 'manifest.json', session_dir / 'import_report.json',
                 *sorted((session_dir / 'collection').glob('*/manifest.json'))]:
        if path.exists():
            data = transform(_read(path))
            if path.name == 'import_report.json':
                path.write_text(json.dumps(data, indent=2) + "\n", encoding='utf-8')
            else:
                _write(path, data)


def rename_session(session_dir: Path, experiment_id: str | None = None, group_id: str | None = None,
                   log=print) -> Path:
    """the session moved to <experiment>_<group>_<start>, ids and paths inside its manifests following"""
    parts = split_session_id(session_dir.name)
    new_id = f"{experiment_id or parts['experiment']}_{group_id or parts['group']}_{parts['start']}"
    if new_id == session_dir.name:
        return session_dir
    target = session_dir.parent / new_id
    if target.exists():
        raise FileExistsError(f"{target} exists")
    log(f"  session {session_dir.name} -> {new_id}")
    shutil.move(str(session_dir), str(target))
    old_root, new_root = str(session_dir), str(target)

    def transform(data):
        data = _replace_paths(data, old_root, new_root)
        if 'session_id' in data:
            data['session_id'] = new_id
        if 'experiment_id' in data or experiment_id:
            data['experiment_id'] = experiment_id or data.get('experiment_id')
        if 'group_id' in data or group_id:
            data['group_id'] = group_id or data.get('group_id')
        return data
    _rewrite_manifests(target, transform)
    return target


def relabel_host(session_dir: Path, old: str, new: str, log=print) -> int:
    """collection/<old> and its recordings renamed to <new>, in the file names and the manifests;
    returns how many files were renamed"""
    old_dir, new_dir = session_dir / 'collection' / old, session_dir / 'collection' / new
    if not old_dir.is_dir():
        raise FileNotFoundError(f"no host {old} in {session_dir.name}")
    if new_dir.exists():
        raise FileExistsError(f"{new_dir} exists")
    log(f"  host {old} -> {new}")
    shutil.move(str(old_dir), str(new_dir))
    renamed = 0
    prefixes = (f"audio_{old}_", f"video_{old}_")
    for path in sorted(new_dir.rglob('*')):
        if path.is_file() and path.name.startswith(prefixes):
            path.rename(path.with_name(path.name.replace(f"_{old}_", f"_{new}_", 1)))
            renamed += 1

    def transform(data):
        data = _replace_paths(data, str(old_dir), str(new_dir))
        # the file names and the recording ids carry the host between the kind and the channel
        text = json.dumps(data)
        for kind in ('audio', 'video'):
            text = text.replace(f"{kind}_{old}_", f"{kind}_{new}_")
        data = json.loads(text)

        def fix_host(value):
            if isinstance(value, dict):
                if value.get('host') == old:
                    value['host'] = new
                for v in value.values():
                    fix_host(v)
            elif isinstance(value, list):
                for v in value:
                    fix_host(v)
        fix_host(data)
        return data
    _rewrite_manifests(session_dir, transform)
    return renamed


def prune_legacy(session_dir: Path, host_map: dict[str, str] | None = None, log=print) -> dict[str, Any]:
    """speaker profiles and meta.txt kept, the rest of legacy/ deleted"""
    legacy = session_dir / 'legacy'
    result: dict[str, Any] = {'profiles': [], 'kept': [], 'deleted_files': 0}
    if not legacy.is_dir():
        return result
    from openmmla.commands.ses.imp import _safe
    for profiles in sorted(p for p in legacy.rglob('profiles') if p.is_dir()):
        base = _safe(profiles.parent.name)
        host = (host_map or {}).get(base, base)
        target = session_dir / 'collection' / host / 'profiles'
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            for child in profiles.iterdir():
                shutil.move(str(child), str(target / child.name))
        else:
            shutil.move(str(profiles), str(target))
        result['profiles'].append(str(target))
        log(f"  profiles of {profiles.parent.name} -> collection/{host}/profiles")
    for name in KEEP_IN_LEGACY_ROOT:
        for path in legacy.rglob(name):
            kept = session_dir / name
            if not kept.exists():
                shutil.move(str(path), str(kept))
                result['kept'].append(str(kept))
    result['deleted_files'] = sum(len(files) for _, _, files in os.walk(legacy))
    shutil.rmtree(legacy)
    log(f"  legacy/ deleted ({result['deleted_files']} files)")
    return result


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mmla ses-tidy',
        description="Rename a session under artifacts/, relabel its hosts, and reduce legacy/ to what is raw.")
    parser.add_argument('session', help="the session folder under artifacts/, or its id")
    parser.add_argument('-e', '--experiment', default=None, help="new experiment id")
    parser.add_argument('-g', '--group', default=None, help="new group id")
    parser.add_argument('--host', action='append', default=[], metavar='OLD=NEW', help="relabel a host (repeatable)")
    parser.add_argument('--prune-legacy', action='store_true', help="keep speaker profiles and meta.txt, delete the rest of legacy/")
    parser.add_argument('-a', '--artifacts', default=None, help="artifacts root (default <cwd>/artifacts)")
    return parser


def main(argv=None):
    args = get_parser().parse_args(argv)
    session_dir = Path(args.session).expanduser()
    if not session_dir.is_dir():
        session_dir = Path(args.artifacts or os.path.join(os.getcwd(), 'artifacts')) / args.session
    if not session_dir.is_dir():
        print(f"no session at {session_dir}")
        return 1
    session_dir = session_dir.resolve()
    host_map = {}
    for pair in args.host:
        old, _, new = pair.partition('=')
        if not old or not new:
            print(f"--host wants OLD=NEW, not {pair!r}")
            return 1
        host_map[old] = new
    print(session_dir.name)
    for old, new in host_map.items():
        renamed = relabel_host(session_dir, old, new)
        print(f"    {renamed} files renamed")
    if args.prune_legacy:
        prune_legacy(session_dir, host_map)
    if args.experiment or args.group:
        session_dir = rename_session(session_dir, args.experiment, args.group)
    print(f"-> {session_dir}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
