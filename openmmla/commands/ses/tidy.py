"""mmla ses-tidy: a session under artifacts/ renamed, its hosts relabelled, a video flipped or a
quadrant of a mosaic cut out, and its legacy folder reduced to what is raw.

The collection layout names a session <experiment>_<group>_<start> and every recording after
its machine and its device: <kind>_<host>_<device>_<start>.<ext>, a single channel of a
multi-channel device appended to the device (vimo-0-ch1). Whatever
changes here, files, folders and manifests move together: the manifests are rebuilt from the
tree at the end (a host's recordings under collection/<host>/{audio,video}/, raw/ holds what is
kept but not replayed), so a relabel is a move and nothing else. --prune-legacy keeps a
session's speaker profiles (as collection/<host>/profiles/) and its meta.txt, and deletes the
rest of legacy/ and the folders an earlier run's analysis produced.

Every audio recording of the manifests says whose voice it holds (`scope`: personal or group)
and, for a worn microphone, who wore it (`participant`, a tag id): what the Collection form's
Participant noted is kept, the device name gives the scope otherwise, and --scope, --participant
and --participants-in-order set them. --participants-in-order hands the session group's tag ids
in config/experiments.yaml, lowest first, to the personal microphones in natural device order,
and 0, 1, 2 ... when the file does not list the group; a microphone already bound to one of those
tags keeps it, and any beyond the list is left unbound.

--same-class-as marks two sessions of different pupils from one school class: each manifest lists
the other under `same_class_as`, which the interaction classifier's folds keep together (the
2025-05-13 takes carry it too). The links survive every rebuild and follow a renamed session.

--pupils declares who the pupils of a session were, as tag ids (`pupils` in the manifest, e.g.
["0", "1"]). The interaction classifier's roster takes them in place of its rules, so a spare badge
lying on the table for a few seconds is not counted as a third pupil. --pupils '' or none clears
them; they survive every rebuild, and each change is noted in the manifest.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

SESSION_ID_RE = re.compile(r'^(?P<experiment>.+?)_(?P<group>group_[^_]+)_(?P<start>\d{6}T\d{4}Z)$')
RECORDING_RE = re.compile(r'^(?P<modality>audio|video)_(?P<host>.+?)_(?P<device>[^_]+)_(?P<start>\d+(?:\.\d+)?)\.(?P<ext>\w+)$')
KEEP_IN_LEGACY_ROOT = ('meta.txt',)
OUTPUT_FOLDERS = ('legacy', 'analysis', 'exports', 'measurements', 'pipelines', 'visualizations', '.staging')
MEDIA_EXTS = ('.wav', '.mp4', '.mov', '.mkv', '.m4a', '.avi', '.webm', '.flac', '.mp3')
CLUTTER = ('.DS_Store', 'Thumbs.db', '.manifest.lock', 'manifest.json', 'manifest.yml')  # never a reason to keep a folder
MAX_PUPILS = 3  # the interaction classifier's slots (layout.N_SLOTS)
KEPT_KEYS = ('legacy_meta', 'imported_from', 'tag_size', 'same_class_as', 'origin_session', 'pupils')  # session manifest keys a rebuild keeps


def _remove_if_empty(folder: Path) -> bool:
    """a folder holding nothing but manifests and Finder droppings is removed"""
    if not folder.is_dir():
        return False
    for path in folder.rglob('*'):
        if path.is_file() and path.name not in CLUTTER:
            return False
    shutil.rmtree(folder)
    return True


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


def channel_of_device(device: str) -> str:
    """the channel a device slot says it is: the -chN suffix of a multi-channel
    device (vimo-0-ch1 -> ch1), else mono. The live recorders write the same."""
    channel = re.search(r'-(ch\d+)$', str(device or ''))
    return channel.group(1) if channel else 'mono'


def parse_recording_name(name: str) -> dict[str, Any] | None:
    match = RECORDING_RE.match(name)
    if not match:
        return None
    parts = match.groupdict()
    parts['start'] = float(parts['start'])
    return parts


def _recording_keys(record: dict[str, Any]) -> set[tuple]:
    """what a manifest entry survives a relabel by: the modality, the start and
    the device slot of its name. A manifest written before the device slot was
    the device kept that slot under `channel` or `camera_label` (`device` was
    the ffmpeg one then), so every spelling is offered and any of them finds it."""
    modality = record.get('modality')
    start = round(float(record.get('start_time') or 0), 3)
    slots = (record.get('device'), record.get('channel'), record.get('camera_label'))
    return {(modality, start, slot) for slot in slots if slot}


def audio_scope_of(record: dict) -> str | None:
    """a manifest audio record's scope: its own when it says personal or group, else the default
    of its device (default_audio_scope, with its import method and host)."""
    from openmmla.collection.recording import AUDIO_SCOPES, default_audio_scope
    scope = record.get('scope')
    if scope in AUDIO_SCOPES:
        return scope
    return default_audio_scope(record.get('device'), (record.get('imported') or {}).get('method'), record.get('host'))


def parse_device_values(values, what: str, allowed: tuple[str, ...] | None = None
                        ) -> list[tuple[str | None, str, str | None]]:
    """[HOST/]DEVICE=VALUE flags as (host or None, device, value); a VALUE of none (any case) is
    None; raises ValueError naming `what` for a malformed one or a value outside `allowed`."""
    wanted = f"[HOST/]DEVICE={'|'.join(allowed)}" if allowed else "[HOST/]DEVICE=VALUE"
    parsed = []
    for value in values or ():
        host_device, _, text = str(value).partition('=')
        host, _, device = host_device.rpartition('/')
        text = text.strip()
        if not device or not text:
            raise ValueError(f"{what} wants {wanted}, not {value!r}")
        if text.lower() == 'none':
            parsed.append((host or None, device, None))
            continue
        if allowed and text not in allowed:
            raise ValueError(f"{what} wants {wanted}, not {text!r}")
        parsed.append((host or None, device, text))
    return parsed


def _experiments_root(session_dir: Path) -> str | None:
    """the project root an artifacts/<session> folder sits in, when its config/experiments.yaml is
    there; None to read the repository's own."""
    root = session_dir.parent.parent
    return str(root) if (root / 'config' / 'experiments.yaml').is_file() else None


def group_tags(experiment_id: str, group_id: str, project_root: str | None = None) -> list[str]:
    """the tag ids of a group's participants in config/experiments.yaml, lowest first (tag 10
    after tag 9); [] for a group the file does not list"""
    from openmmla.collection.recording import participant_roster
    from openmmla.utils.experiments import load_experiments
    return [tag for _, tag in participant_roster(load_experiments(project_root), experiment_id, group_id)]


def assign_audio_roles(records: list[dict], participants=(), scopes=(), in_order: bool = False,
                       session: str = '', log=print, order_tags: list[str] | None = None,
                       order_group: str = '') -> None:
    """give every audio record its scope and participant, in place: the defaults first (a record
    keeps a valid scope it has), then --scope, then tags in natural device order for the personal
    ones (--participants-in-order: `order_tags`, the group's tags lowest first, else 0, 1, 2 ...;
    a microphone already bound to one of them keeps it, the others take the rest one each and any
    beyond them are left unbound, and every binding it changes is said), then --participant; a
    participant makes a record personal, and a group record has none. A flag that names no audio
    of the session says so."""
    from openmmla.collection.recording import natural_device_key

    def matching(host, device):
        found = [r for r in records if r.get('device') == device and (host is None or r.get('host') == host)]
        if not found:
            log(f"    [no audio of {session} records {f'{host}/' if host else ''}{device}]")
        return found

    for record in records:
        record['scope'] = audio_scope_of(record)
        record.setdefault('participant', None)
    for host, device, scope in scopes:
        for record in matching(host, device):
            record['scope'] = scope
    if in_order:
        pairs = sorted({(r.get('host'), r.get('device')) for r in records if r.get('scope') == 'personal'},
                       key=lambda pair: (natural_device_key(pair[1]), str(pair[0])))
        order = [str(tag) for tag in order_tags] if order_tags else [str(i) for i in range(len(pairs))]
        bound: dict[tuple, str] = {}
        for record in records:
            pair = (record.get('host'), record.get('device'))
            if pair in pairs and record.get('participant') is not None:
                bound.setdefault(pair, str(record['participant']))
        # a microphone already bound to one of the tags keeps it (the Collection form's pick, an
        # earlier --participant); the others take the tags left, in natural device order
        tags: dict[tuple, str | None] = {}
        for pair in pairs:
            if bound.get(pair) in order and bound[pair] not in tags.values():
                tags[pair] = bound[pair]
        free = [tag for tag in order if tag not in tags.values()]
        rest = [pair for pair in pairs if pair not in tags]
        for i, pair in enumerate(rest):
            tags[pair] = free[i] if i < len(free) else None
        kept = [f"{device} tag {tags[(host, device)]}" for host, device in pairs if (host, device) not in rest]
        if kept:
            log(f"    [{session}: already bound, kept: {', '.join(kept)}]")
        for host, device in rest:
            before, after = bound.get((host, device)), tags[(host, device)]
            if before is not None and before != after:
                log(f"    [{session}: {device} was tag {before}, now {f'tag {after}' if after else 'unbound'}]")
        if len(rest) > len(free):
            devices = [device for _, device in rest[len(free):]]
            log(f"    [{session}: more personal microphones than {order_group or 'the group'} has tags: "
                f"{', '.join(devices)} left unbound]")
        for record in records:
            pair = (record.get('host'), record.get('device'))
            if pair in tags:
                record['participant'] = tags[pair]
    for host, device, tag in participants:
        for record in matching(host, device):
            record['participant'] = tag
            if tag is not None:
                record['scope'] = 'personal'  # a wearer makes it one person's microphone
    for record in records:
        if record.get('scope') == 'group':
            record['participant'] = None
    holders: dict[str, list[tuple]] = {}
    for record in records:
        tag = record.get('participant')
        pair = (record.get('host'), record.get('device'))
        if tag is not None and pair not in holders.setdefault(tag, []):
            holders[tag].append(pair)
    for tag, pairs in holders.items():
        if len(pairs) > 1:
            log(f"    [{session}: tag {tag} is on both {pairs[0][1]} and {pairs[1][1]}]")


def _moved_entries(known: dict[tuple, list[dict[str, Any]]], present: set[tuple]) -> list[dict[str, Any]]:
    """the old manifest entries no file of the tree starts at any more (ses-align renamed it to a
    new start), one per recording"""
    moved, seen = [], set()
    for entries in known.values():
        for r in entries:
            keys = _recording_keys(r)
            ident = (r.get('host'), r.get('modality'), round(float(r.get('start_time') or 0), 3), r.get('device'))
            if keys & present or ident in seen:
                continue
            seen.add(ident)
            moved.append(r)
    return moved


def _claim_moved(moved: list[dict[str, Any]], parsed: dict[str, Any], host: str) -> dict[str, Any]:
    """the moved entry of this recording, taken out of `moved`: the same kind and device slot, this
    machine's before another's, the nearest start first; its duration and end left to the probe
    (a trim changed them). {} when none is left."""
    slots = lambda r: {r.get('device'), r.get('channel'), r.get('camera_label')}
    fits = [r for r in moved if r.get('modality') == parsed['modality'] and parsed['device'] in slots(r)]
    if not fits:
        return {}
    best = min(fits, key=lambda r: (r.get('host') != host, abs(float(r.get('start_time') or 0) - parsed['start'])))
    moved.remove(best)
    return {k: v for k, v in best.items() if k not in ('duration', 'stopped_at')}


def rebuild_manifests(session_dir: Path, experiment_id: str | None = None, group_id: str | None = None,
                      notes: list[str] | None = None, log=print, participants=None, scopes=None,
                      participants_in_order: bool = False,
                      order_group: tuple[str | None, str | None] = (None, None),
                      project_root: str | None = None) -> dict[str, Any]:
    """the host manifests and the session manifest written again from what collection/ holds,
    keeping what the old manifests knew about each recording (duration, how it was imported, its
    scope and participant); `participants`, `scopes` and `participants_in_order` set the scope and
    wearer of audio recordings (assign_audio_roles), in order with the tags of the group
    `order_group` names (-e/-g), else the session's, in the experiments.yaml of `project_root`
    (default: the one beside artifacts/, else the repository's)"""
    from openmmla.collection.recording import format_epoch_ms
    from openmmla.commands.ses.imp import probe

    old = _read(session_dir / 'manifest.json')
    known: dict[tuple, list[dict[str, Any]]] = {}
    manifests = [old] + [_read(path) for path in sorted((session_dir / 'collection').glob('*/manifest.json'))]
    for manifest in manifests:
        for r in manifest.get('recordings', []):
            if isinstance(r, dict):
                for key in _recording_keys(r):
                    known.setdefault(key, []).append(r)
    parts = split_session_id(session_dir.name)
    session_id = session_dir.name
    now = format_epoch_ms()
    files = []
    for path in sorted((session_dir / 'collection').glob('*/*/*')):
        if not path.is_file() or path.parent.name not in ('audio', 'video') or path.suffix.lower() not in MEDIA_EXTS:
            continue
        parsed = parse_recording_name(path.name)
        if not parsed:
            log(f"  {path.relative_to(session_dir)}: not <kind>_<host>_<device>_<start>.<ext>, left out of the manifest")
            continue
        files.append((path, parsed))
    moved = _moved_entries(known, {(parsed['modality'], round(parsed['start'], 3), parsed['device']) for _, parsed in files})
    hosts: dict[str, list[dict[str, Any]]] = {}
    for path, parsed in files:
        host = path.parent.parent.name
        # the entry of this machine when two recorded the same device at the same moment, else the
        # first (a relabelled host's entry still names the old one)
        candidates = known.get((parsed['modality'], round(parsed['start'], 3), parsed['device']), [])
        record = dict(next((r for r in candidates if r.get('host') == host), candidates[0] if candidates else {}))
        if not candidates:
            # a recording whose start moved (ses-align shifted or trimmed it) keeps what its old
            # entry knew, its length probed again
            record = _claim_moved(moved, parsed, host)
        if not record.get('duration'):
            record['duration'] = round(float(probe(str(path)).get('duration') or 0), 3) or None
        record.update({'id': path.stem, 'modality': parsed['modality'], 'status': 'stopped', 'path': str(path),
                       'start_time': round(parsed['start'], 3), 'host': host, 'format': parsed['ext']})
        # the device slot of the name: the physical device and its base id (c920-01, vimo-0,
        # badge-0, jabra-0), a channel of a multi-channel device as a -chN suffix
        record['device'] = parsed['device']
        if parsed['modality'] == 'audio':
            record['channel'] = channel_of_device(parsed['device'])
            record.setdefault('channels', 1)
            record.setdefault('sample_rate', 16000)
        else:
            record.pop('channel', None)
        if record.get('duration'):
            record['stopped_at'] = round(record['start_time'] + record['duration'], 3)
        hosts.setdefault(host, []).append(record)

    all_records = [r for records in hosts.values() for r in records]
    order_tags: list[str] = []
    order_name = ''
    if participants_in_order:
        experiment = order_group[0] or experiment_id or old.get('experiment_id') or parts['experiment']
        group = order_group[1] or group_id or old.get('group_id') or parts['group']
        order_name = f"{experiment}/{group}"
        order_tags = group_tags(experiment, group, project_root or _experiments_root(session_dir))
        if order_tags:
            log(f"    [{session_id}: personal microphones tagged in natural device order with {experiment}/{group}'s "
                f"tags {', '.join(order_tags)}]")
        else:
            log(f"    [{session_id}: {experiment}/{group} has no participants in config/experiments.yaml: "
                f"personal microphones tagged 0, 1, 2 ... in natural device order]")
    # the same dicts as the host manifests', so both say the same
    assign_audio_roles([r for r in all_records if r['modality'] == 'audio'], participants or (), scopes or (),
                       participants_in_order, session_id, log, order_tags=order_tags, order_group=order_name)
    sync = max((r['start_time'] for r in all_records), default=float(old.get('initial_sync_time') or 0))
    for host, records in hosts.items():
        host_dir = session_dir / 'collection' / host
        _write(host_dir / 'manifest.json', {
            'session_id': session_id, 'initial_sync_time': sync,
            'created_at': _read(host_dir / 'manifest.json').get('created_at') or format_epoch_ms(sync),
            'updated_at': now, 'recordings': records})
    for stale in (session_dir / 'collection').glob('*/manifest.json'):
        if stale.parent.name not in hosts:
            stale.unlink()
            stale.with_suffix('.yml').unlink(missing_ok=True)

    file_sources: dict[str, list[dict[str, Any]]] = {'asr': [], 'ips': [], 'vfa': []}
    for host, records in hosts.items():
        host_dir = session_dir / 'collection' / host
        if any(r['modality'] == 'audio' for r in records):
            file_sources['asr'].append({'host': host, 'pipeline': 'collection', 'source': 'file',
                                        'file_dir': str(host_dir / 'audio'), 'initial_sync_time': sync})
        if any(r['modality'] == 'video' for r in records):
            for pipeline in ('ips', 'vfa'):
                file_sources[pipeline].append({'host': host, 'pipeline': 'collection', 'source': 'file',
                                               'file_dir': str(host_dir / 'video'), 'initial_sync_time': sync})
    data: dict[str, Any] = {
        'session_id': session_id,
        'experiment_id': experiment_id or old.get('experiment_id') or parts['experiment'],
        'group_id': group_id or old.get('group_id') or parts['group'],
        'initial_sync_time': sync,
        'created_at': old.get('created_at') or format_epoch_ms(sync),
        'updated_at': now,
        'artifacts': {'collection': [{'host': host, 'local_path': str(session_dir / 'collection' / host), 'updated_at': now}
                                     for host in hosts]},
        'file_sources': {k: v for k, v in file_sources.items() if v},
        'recordings': all_records,
    }
    for key in KEPT_KEYS:
        if old.get(key) is not None:
            data[key] = old[key]
    raw = session_dir / 'raw'
    if raw.is_dir():
        data['raw'] = sorted(str(p.relative_to(session_dir)) for p in raw.rglob('*') if p.is_file())
    kept_notes = [n for n in (old.get('notes') or []) if isinstance(n, str)]
    for note in notes or []:
        if note not in kept_notes:
            kept_notes.append(note)
    if kept_notes:
        data['notes'] = kept_notes
    _write(session_dir / 'manifest.json', data)
    return data


def rename_session(session_dir: Path, experiment_id: str | None = None, group_id: str | None = None,
                   log=print) -> Path:
    """the session moved to <experiment>_<group>_<start>, its manifests rebuilt on the new paths"""
    parts = split_session_id(session_dir.name)
    new_id = f"{experiment_id or parts['experiment']}_{group_id or parts['group']}_{parts['start']}"
    if new_id == session_dir.name:
        rebuild_manifests(session_dir, experiment_id, group_id, log=log)
        return session_dir
    target = session_dir.parent / new_id
    if target.exists():
        raise FileExistsError(f"{target} exists")
    log(f"  session {session_dir.name} -> {new_id}")
    shutil.move(str(session_dir), str(target))
    _relink(target, session_dir.name, log=log)
    report = target / 'import_report.json'
    if report.exists():
        data = _read(report)
        data['target'] = str(target)
        data['session_id'] = new_id
        report.write_text(json.dumps(data, indent=2) + "\n", encoding='utf-8')
    rebuild_manifests(target, experiment_id or parts['experiment'], group_id or parts['group'], log=log)
    return target


def parse_pupils(value: str) -> list[str] | None:
    """--pupils as the manifest keeps it: tag ids as text, in the order given ('0,1' -> ['0', '1']);
    None for '' or none, which clears them. Raises ValueError on anything but distinct tag ids, or
    more than MAX_PUPILS of them (the classifier holds at most that many persons)."""
    text = value.strip()
    if text.lower() in ('', 'none'):
        return None
    tags = [part.strip() for part in text.split(',')]
    if any(not tag.isdigit() for tag in tags):
        raise ValueError(f"--pupils wants tag ids separated by commas (0,1), or none, not {value!r}")
    tags = [str(int(tag)) for tag in tags]
    if len(set(tags)) != len(tags):
        raise ValueError(f"--pupils names a tag twice: {value!r}")
    if len(tags) > MAX_PUPILS:
        raise ValueError(f"--pupils names {len(tags)} tags; the classifier holds at most {MAX_PUPILS} persons")
    return tags


def set_pupils(session_dir: Path, pupils: list[str] | None) -> str | None:
    """the session manifest's `pupils` set to `pupils` (None removes them); returns the note that
    says so, or None when nothing changed."""
    path = session_dir / 'manifest.json'
    data = _read(path)
    if (data.get('pupils') or None) == (list(pupils) if pupils else None):
        return None
    # dated, so setting the same pupils again after a clear is noted again, in order
    stamp = time.strftime('%Y-%m-%dT%H:%MZ', time.gmtime())
    if pupils:
        data['pupils'] = list(pupils)
        note = f"{stamp} pupils: tags {', '.join(pupils)} (ses-tidy --pupils; the classifier's roster takes them in place of its rules)"
    else:
        data.pop('pupils', None)
        note = f"{stamp} pupils cleared (ses-tidy --pupils none; the classifier's roster rules apply again)"
    _write(path, data)
    return note


def _same_class(data: dict[str, Any]) -> list[str]:
    linked = data.get('same_class_as') or []
    return [linked] if isinstance(linked, str) else [str(s) for s in linked if s]


def link_same_class(session_dir: Path, other: str | Path, log=print) -> Path:
    """two sessions of different pupils from one school class marked as such: each manifest lists
    the other under same_class_as (symmetric; a link already there is kept once). `other` is a
    session folder or an id beside this one. Returns the other session's folder."""
    other_dir = Path(other).expanduser()
    if not other_dir.is_dir():
        other_dir = session_dir.parent / str(other)
    if not (other_dir / 'manifest.json').is_file():
        raise FileNotFoundError(f"no session manifest at {other_dir}")
    other_dir = other_dir.resolve()
    if other_dir == session_dir.resolve():
        raise ValueError("a session is not the same class as itself")
    for here, there in ((session_dir, other_dir), (other_dir, session_dir)):
        path = here / 'manifest.json'
        data = _read(path)
        linked = _same_class(data)
        if there.name not in linked:
            data['same_class_as'] = linked + [there.name]
            _write(path, data)
        log(f"  {here.name}: same_class_as {', '.join(_same_class(data))}")
    return other_dir


def _relink(session_dir: Path, old_id: str, log=print) -> None:
    """the sessions a renamed one is linked to name it by its new id"""
    for other in _same_class(_read(session_dir / 'manifest.json')):
        path = session_dir.parent / other / 'manifest.json'
        data = _read(path)
        linked = _same_class(data)
        if old_id in linked:
            data['same_class_as'] = [session_dir.name if s == old_id else s for s in linked]
            _write(path, data)
            log(f"  {other}: same_class_as follows the rename to {session_dir.name}")


def _rename_in_place(path: Path, old_host: str, new_host: str) -> Path:
    parsed = parse_recording_name(path.name)
    if parsed and parsed['host'] == old_host:
        target = path.with_name(f"{parsed['modality']}_{new_host}_{parsed['device']}_{path.name.split('_', 2)[2].split('_', 1)[1]}")
        path.rename(target)
        return target
    return path


def relabel_host(session_dir: Path, old: str, new: str, modality: str | None = None, log=print) -> int:
    """collection/<old> (or only its audio or video) moved to collection/<new>, its files renamed;
    returns how many files moved"""
    old_dir, new_dir = session_dir / 'collection' / old, session_dir / 'collection' / new
    if not old_dir.is_dir():
        raise FileNotFoundError(f"no host {old} in {session_dir.name}")
    folders = [modality] if modality else ['audio', 'video', 'profiles']
    moved = 0
    log(f"  {modality or 'host'} {old} -> {new}")
    for folder in folders:
        source = old_dir / folder
        if not source.is_dir():
            continue
        target = new_dir / folder
        target.mkdir(parents=True, exist_ok=True)
        for path in sorted(source.iterdir()):
            destination = target / path.name
            if destination.exists():
                raise FileExistsError(f"{destination} exists")
            shutil.move(str(path), str(destination))
            if path.is_file() or destination.is_file():
                _rename_in_place(destination, old, new)
                moved += 1
        _remove_if_empty(source)
    _remove_if_empty(old_dir)
    return moved


def relabel_device(session_dir: Path, old: str, new: str, host: str | None = None, log=print) -> int:
    """the device slot of a session's file names, `old` -> `new`, under every
    host or only under `host`; returns how many files were renamed.

    The slot is taken as it is spelled, channel suffix and all, so ses-import's
    placeholders become the device that recorded: `video0` -> `c920-01`,
    `ch1` -> `vimo-0-ch1`. Nothing moves between folders: the machine is the
    folder (relabel_host) and the device is in the name."""
    from openmmla.collection.recording import default_audio_scope
    collection = session_dir / 'collection'
    hosts = [collection / host] if host else sorted(p for p in collection.glob('*') if p.is_dir())
    if host and not hosts[0].is_dir():
        raise FileNotFoundError(f"no host {host} in {session_dir.name}")
    log(f"  device {old} -> {new}" + (f" on {host}" if host else ""))
    renamed = 0
    for host_dir in hosts:
        for folder in ('audio', 'video'):
            source = host_dir / folder
            if not source.is_dir():
                continue
            for path in sorted(source.iterdir()):
                parsed = parse_recording_name(path.name) if path.is_file() else None
                if not parsed or parsed['device'] != old:
                    continue
                target = path.with_name(
                    f"{parsed['modality']}_{parsed['host']}_{new}_{parsed['start']:.3f}.{parsed['ext']}")
                if target.exists():
                    raise FileExistsError(f"{target} exists")
                path.rename(target)
                renamed += 1
    if not renamed:
        log(f"    [no file of {session_dir.name} records {old}]")
        return 0
    # the manifests move with the files: a rebuild finds an entry again by its
    # start and its device slot, so the slot it knows has to change as well
    for manifest in [session_dir / 'manifest.json', *sorted((session_dir / 'collection').glob('*/manifest.json'))]:
        data = _read(manifest)
        entries = [
            r for r in data.get('recordings', [])
            if isinstance(r, dict) and old in (r.get('device'), r.get('channel'), r.get('camera_label'))
        ]
        if host:
            entries = [r for r in entries if r.get('host') == host]
        if not entries:
            continue
        for record in entries:
            if record.get('modality') == 'audio' and record.get('scope') == default_audio_scope(
                    old, (record.get('imported') or {}).get('method'), record.get('host')):
                record.pop('scope', None)  # only the old name's default: the rebuild works it out for the new name
            record['device'] = new
            record.pop('camera_label', None)
            if record.get('modality') == 'audio':
                record['channel'] = channel_of_device(new)
        _write(manifest, data)
    return renamed


def move_to_raw(session_dir: Path, host: str, log=print) -> Path:
    """collection/<host> kept under raw/<host>: not a source any more, not deleted either"""
    source = session_dir / 'collection' / host
    if not source.is_dir():
        raise FileNotFoundError(f"no host {host} in {session_dir.name}")
    target = session_dir / 'raw' / host
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        raise FileExistsError(f"{target} exists")
    for name in ('manifest.json', 'manifest.yml', '.manifest.lock'):
        (source / name).unlink(missing_ok=True)
    shutil.move(str(source), str(target))
    log(f"  host {host} -> raw/{host}")
    return target


def delete_host(session_dir: Path, host: str, log=print) -> int:
    source = session_dir / 'collection' / host
    if not source.is_dir():
        raise FileNotFoundError(f"no host {host} in {session_dir.name}")
    count = sum(len(files) for _, _, files in os.walk(source))
    shutil.rmtree(source)
    log(f"  host {host} deleted ({count} files)")
    return count


def _encoder_args(source: str) -> list[str]:
    """the h264 encoder: VideoToolbox on a Mac at the source's bitrate, libx264 elsewhere"""
    from openmmla.commands.ses.imp import probe
    info = probe(source)
    size = os.path.getsize(source)
    bitrate = int(size * 8 / (info.get('duration') or 1))
    bitrate = max(2_000_000, min(bitrate, 12_000_000))
    encoders = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
    if 'h264_videotoolbox' in encoders:
        return ['-c:v', 'h264_videotoolbox', '-b:v', str(bitrate), '-pix_fmt', 'yuv420p']
    return ['-c:v', 'libx264', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p']


def transcode(source: str, destination: str, video_filter: str, fps: int | None = None, keep_audio: bool = True) -> None:
    command = ['ffmpeg', '-v', 'error', '-y', '-i', source, '-vf', video_filter, *_encoder_args(source)]
    if fps:
        command += ['-r', str(fps)]
    command += ['-c:a', 'aac', '-b:a', '128k'] if keep_audio else ['-an']
    command += ['-movflags', '+faststart', destination]
    subprocess.run(command, check=True, timeout=6 * 3600)


def flip_video(session_dir: Path, host: str, log=print) -> list[str]:
    """every video of the host turned by 180 degrees (re-encoded); the original kept under raw/"""
    video_dir = session_dir / 'collection' / host / 'video'
    done = []
    for path in sorted(video_dir.glob('video_*')):
        if path.suffix.lower() not in MEDIA_EXTS:
            continue
        log(f"  flipping {path.name}")
        flipped = path.with_suffix('.mp4')
        temp = path.with_name(f".{path.stem}.flipping.mp4")
        transcode(str(path), str(temp), 'hflip,vflip')
        raw_dir = session_dir / 'raw' / host / 'upside-down'
        raw_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(raw_dir / path.name))
        temp.rename(flipped)
        done.append(flipped.name)
    return done


def crop_video(session_dir: Path, host: str, new_host: str, x: int, y: int, w: int, h: int, fps: int | None = None,
               log=print) -> list[str]:
    """a region of every video of the host cut out into a video of new_host, same start, no audio"""
    video_dir = session_dir / 'collection' / host / 'video'
    target_dir = session_dir / 'collection' / new_host / 'video'
    target_dir.mkdir(parents=True, exist_ok=True)
    done = []
    for path in sorted(video_dir.glob('video_*')):
        parsed = parse_recording_name(path.name)
        if not parsed or path.suffix.lower() not in MEDIA_EXTS:
            continue
        target = target_dir / f"video_{new_host}_{parsed['device']}_{parsed['start']:.3f}.mp4"
        log(f"  cropping {path.name} [{w}x{h} at {x},{y}] -> {new_host}")
        temp = target.with_name(f".{target.stem}.cropping.mp4")
        transcode(str(path), str(temp), f'crop={w}:{h}:{x}:{y}', fps=fps, keep_audio=False)
        temp.rename(target)
        done.append(target.name)
    return done


def prune_legacy(session_dir: Path, log=print) -> dict[str, Any]:
    """speaker profiles and meta.txt kept, legacy/ and the old analysis folders deleted"""
    from openmmla.commands.ses.imp import _safe
    result: dict[str, Any] = {'profiles': [], 'kept': [], 'deleted_files': 0, 'deleted_folders': []}
    legacy = session_dir / 'legacy'
    if legacy.is_dir():
        for profiles in sorted(p for p in legacy.rglob('profiles') if p.is_dir()):
            host = _safe(profiles.parent.name)
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
    for folder in OUTPUT_FOLDERS:
        target = session_dir / folder
        if target.is_dir():
            result['deleted_files'] += sum(len(files) for _, _, files in os.walk(target))
            shutil.rmtree(target)
            result['deleted_folders'].append(folder)
    if result['deleted_folders']:
        log(f"  deleted {', '.join(result['deleted_folders'])} ({result['deleted_files']} files)")
    return result


def get_parser():
    parser = argparse.ArgumentParser(
        prog='mmla ses-tidy',
        description="Rename a session under artifacts/, relabel its hosts, flip or crop its videos, "
                    "and reduce legacy/ to what is raw. The manifests are rebuilt from the tree afterwards.")
    parser.add_argument('session', help="the session folder under artifacts/, or its id")
    parser.add_argument('-e', '--experiment', default=None, help="new experiment id")
    parser.add_argument('-g', '--group', default=None, help="new group id")
    parser.add_argument('--host', action='append', default=[], metavar='OLD=NEW', help="relabel a host, audio and video (repeatable)")
    parser.add_argument('--audio-host', action='append', default=[], metavar='OLD=NEW', help="move only the audio of a host to another label")
    parser.add_argument('--video-host', action='append', default=[], metavar='OLD=NEW', help="move only the video of a host to another label")
    parser.add_argument('--device', action='append', default=[], metavar='[HOST/]OLD=NEW',
                        help="rename the device in the file names (video0=c920-01, ch1=vimo-0-ch1); "
                             "HOST/ limits it to one machine's files (repeatable)")
    parser.add_argument('--participant', action='append', default=[], metavar='[HOST/]DEVICE=TAG',
                        help="the participant (tag id) wearing a personal microphone (vimo-0=0, vimo-0-ch1=2); "
                             "none unbinds it (repeatable)")
    parser.add_argument('--participants-in-order', action='store_true',
                        help="tag the personal microphones, in natural device order (vimo-0-ch0 < vimo-0-ch1 "
                             "< vimo-1 < vimo-10), with the tag ids of the session group's participants in "
                             "config/experiments.yaml, lowest first; 0, 1, 2 ... when the group has none; "
                             "one already bound to one of them keeps it; --participant overrides")
    parser.add_argument('--scope', action='append', default=[], metavar='[HOST/]DEVICE=personal|group',
                        help="whether a microphone is one person's or the group's (repeatable)")
    parser.add_argument('--crop', action='append', default=[], metavar='HOST=NEW:X,Y,W,H',
                        help="cut a region of a host's videos out as the videos of NEW (a quadrant of a mosaic)")
    parser.add_argument('--crop-fps', type=int, default=None, help="frame rate of the cropped videos (default: the source's)")
    parser.add_argument('--flip', action='append', default=[], metavar='HOST', help="turn a host's videos by 180 degrees")
    parser.add_argument('--to-raw', action='append', default=[], metavar='HOST', help="keep a host's files under raw/, out of the sources")
    parser.add_argument('--delete-host', action='append', default=[], metavar='HOST', help="delete a host's files")
    parser.add_argument('--tag-size', type=float, default=None, help="the AprilTag size of the session, in metres, noted in the manifest")
    parser.add_argument('--note', action='append', default=[], help="a note kept in the session manifest (repeatable)")
    parser.add_argument('--same-class-as', action='append', default=[], metavar='SESSION',
                        help="another session of different pupils from the same school class: both manifests list each "
                             "other under same_class_as, so the classifier's folds keep them together (repeatable)")
    parser.add_argument('--pupils', default=None, metavar='TAGS',
                        help="the tag ids of the session's pupils, separated by commas (0,1), noted in the manifest; "
                             "the classifier's roster takes them in place of its rules; '' or none clears them")
    parser.add_argument('--prune-legacy', action='store_true', help="keep speaker profiles and meta.txt, delete legacy/ and the old analysis folders")
    parser.add_argument('-a', '--artifacts', default=None, help="artifacts root (default <cwd>/artifacts)")
    return parser


def _pairs(values: list[str], what: str) -> list[tuple[str, str]]:
    pairs = []
    for value in values:
        old, _, new = value.partition('=')
        if not old or not new:
            raise ValueError(f"{what} wants OLD=NEW, not {value!r}")
        pairs.append((old, new))
    return pairs


def main(argv=None):
    args = get_parser().parse_args(argv)
    session_dir = Path(args.session).expanduser()
    if not session_dir.is_dir():
        session_dir = Path(args.artifacts or os.path.join(os.getcwd(), 'artifacts')) / args.session
    if not session_dir.is_dir():
        print(f"no session at {session_dir}")
        return 1
    session_dir = session_dir.resolve()
    try:
        hosts, audio_hosts, video_hosts = (_pairs(v, w) for v, w in ((args.host, '--host'), (args.audio_host, '--audio-host'), (args.video_host, '--video-host')))
        devices = []
        for host_old, new in _pairs(args.device, '--device'):
            host, _, old = host_old.rpartition('/')
            if not old:
                raise ValueError(f"--device wants [HOST/]OLD=NEW, not {host_old}={new!r}")
            devices.append((host or None, old, new))
        crops = []
        for value in args.crop:
            spec, _, region = value.partition(':')
            host, _, new = spec.partition('=')
            numbers = [int(n) for n in region.split(',')] if region else []
            if not host or not new or len(numbers) != 4:
                raise ValueError(f"--crop wants HOST=NEW:X,Y,W,H, not {value!r}")
            crops.append((host, new, numbers))
        from openmmla.collection.recording import AUDIO_SCOPES
        participants = parse_device_values(args.participant, '--participant')
        scopes = parse_device_values(args.scope, '--scope', AUDIO_SCOPES)
        if any(scope is None for _, _, scope in scopes):
            raise ValueError("--scope wants [HOST/]DEVICE=personal|group, not 'none'")
        pupils = parse_pupils(args.pupils) if args.pupils is not None else None
        # checked before anything changes: each named session must be there
        for other in args.same_class_as:
            other_dir = Path(other).expanduser()
            if not other_dir.is_dir():
                other_dir = session_dir.parent / other
            if not (other_dir / 'manifest.json').is_file():
                raise ValueError(f"--same-class-as: no session manifest at {other_dir}")
            if other_dir.resolve() == session_dir:
                raise ValueError("--same-class-as names the session itself")
    except ValueError as error:
        print(error)
        return 1
    print(session_dir.name)
    started = time.time()
    notes = list(args.note)
    # legacy first: its profiles land under the host label the base had, which a relabel then moves
    if args.prune_legacy:
        prune_legacy(session_dir)
    for host, new, (x, y, w, h) in crops:
        made = crop_video(session_dir, host, new, x, y, w, h, fps=args.crop_fps)
        notes.append(f"{new}: {w}x{h} at {x},{y} cut out of {host}'s video")
        print(f"    {len(made)} videos cropped")
    for host in args.flip:
        made = flip_video(session_dir, host)
        notes.append(f"{host}: video turned by 180 degrees (the original under raw/{host}/upside-down)")
        print(f"    {len(made)} videos flipped")
    for old, new in hosts:
        print(f"    {relabel_host(session_dir, old, new)} files moved")
    for old, new in audio_hosts:
        print(f"    {relabel_host(session_dir, old, new, modality='audio')} files moved")
    for old, new in video_hosts:
        print(f"    {relabel_host(session_dir, old, new, modality='video')} files moved")
    # after the host relabels: a device is renamed where its machine's files now are
    for host, old, new in devices:
        print(f"    {relabel_device(session_dir, old, new, host=host)} files renamed")
    for host in args.to_raw:
        move_to_raw(session_dir, host)
    for host in args.delete_host:
        delete_host(session_dir, host)
    if args.tag_size is not None:
        data = _read(session_dir / 'manifest.json')
        data['tag_size'] = args.tag_size
        _write(session_dir / 'manifest.json', data)
    if args.pupils is not None:
        note = set_pupils(session_dir, pupils)
        if note:
            notes.append(note)
            print(f"    {note}")
    rebuild_manifests(session_dir, notes=notes, participants=participants, scopes=scopes,
                      participants_in_order=args.participants_in_order, order_group=(args.experiment, args.group))
    session_dir = rename_session(session_dir, args.experiment, args.group)
    for other in args.same_class_as:
        link_same_class(session_dir, other)
    print(f"-> {session_dir}  ({time.time() - started:.0f} s)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
