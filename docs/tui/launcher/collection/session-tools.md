# Bringing recordings in

Three commands bring recordings made outside the Collection card into a session, and put a session's recordings right: `mmla ses-import` builds the session from a folder of files, `mmla ses-tidy` corrects names, devices and wearers, and `mmla ses-align` puts the recordings on one clock. They work on `artifacts/<session>/`, the same tree the [Collection](index.md) card writes.

## Import recordings

`mmla ses-import` builds the `artifacts/<session>/collection/<host>/{audio,video}/` tree and its manifests from whatever a folder holds.

```bash
mmla ses-import -n "<folder>/session_<ISO>Z"                  # show the plan, change nothing
mmla ses-import -g <group> "<folder>/session_<ISO>Z"
mmla ses-import --only session_<ISO>Z -g <group> "<folder>"   # a folder holding several sessions
```

| Option | Default | What it does |
|---|---|---|
| `-n`, `--dry-run` | off | show the plan and change nothing |
| `-e`, `--experiment` | `exp_wegrow_life` | the experiment id |
| `-g`, `--group` | `group_01` | the group id |
| `-sid`, `--session-id` | `<experiment>_<group>_<start>` | the session id outright; the start comes from a `session_<ISO>Z` folder name |
| `--only` | | when the folder holds several sessions: the `session_<ISO>Z` to import |
| `--tz` | `Europe/Copenhagen` | the time zone of file names that carry a local time |
| `--copy` | off | copy the files and leave the source untouched, instead of moving them |
| `--close-clock-steps` | off | take a clock step out of the stamps instead of filling it with silence (see the clock steps details below) |
| `-a`, `--artifacts` | `<project>/artifacts` | the artifacts root |

### What the import recognizes

| File | Example |
|---|---|
| a video named by the local time it started | `<YYYY-MM-DD> <HH-MM-SS>.mov`, `record_<YYYYMMDD>_<HHMMSS>.mp4` |
| a file named with its unix start | `record_<unix>_<host>.mp4`, `audio_<unix>_ch0.wav` |
| an ASR base's folder of three-second recordings, even downloaded twice | `badge_0/records/badge_0_record_<unix>.wav`, or `segments/` when the raw records were not kept |

Continuous files are moved under their new names, an ASR base's segments are joined into one file, a video's own audio track becomes a 16 kHz group microphone, and everything else in the folder (logs, frames, old outputs, speaker profiles) moves to `legacy/` under the session. `import_report.json` says what went where.

!!! note "A late camera shortens the replay"
    A replay starts at the session's `initial_sync_time`, the latest start among its recordings, so a camera that started late shortens the replay of everything else.

??? info "Details: names, segments and labels"
    - A base's segments are concatenated into one file that starts when the session's first segment did, with silence where nothing was recorded, so every base of a session shares one start.
    - Two videos of one camera recorded one after the other keep one host label.
    - A name carrying two `<label>-<number>` parts gives both slots, machine first and device second: `pi-01_c920-01_<unix>.mov` becomes `video_pi-01_c920-01_<unix>.mov`, and `pi-04_vimo-0_..._ch1.wav` becomes `audio_pi-04_vimo-0-ch1_....wav`. One part alone is the machine, and the device slot keeps a placeholder (`video0`, `mix`, `ch1`, `mic`) for `ses-tidy --device` to correct.

??? info "Details: clock steps"
    - A base stamps each record with its machine's wall clock. A gap that opens at the same moment and lasts as long in every base recording is reported as a clock step: in the plan, in `import_report.json` under `clock_steps`, and in each base's manifest notes.
    - Such a gap is either a step of that machine's clock or all bases stopping at once; only a continuous recording, such as a camera's track, tells the two apart. Check against one before `--close-clock-steps` takes the steps out of the stamps instead of filling them with silence.
    - A gap in one base alone stays a recording gap either way.

## Correct a session

`mmla ses-tidy` corrects an imported session: folders, file names and manifests together. The manifests are rebuilt from the tree at the end.

```bash
mmla ses-tidy <session-id> -e <experiment-id> \
  --crop cam-1=pi-01:0,0,960,540 --crop cam-1=pi-04:960,540,960,540 --crop-fps 30 \
  --device pi-01/video0=c920-01 --device pi-04/video0=c920-04 \
  --audio-host cam-1=jabra --to-raw cam-1 --tag-size 0.07 --prune-legacy
```

| Option | What it does |
|---|---|
| `-e`, `-g` | rename the experiment or the group; the session moves to `<experiment>_<group>_<start>` |
| `--host OLD=NEW` | relabel a host: the machine folder (the device sits in the file names) |
| `--video-host OLD=NEW`, `--audio-host OLD=NEW` | move only one modality; a camera's own audio track becomes the `jabra` group microphone |
| `--device [HOST/]OLD=NEW` | rename the device in the file names, for the placeholders an import leaves (`--device video0=c920-01`, `--device ch1=vimo-0-ch1`); `HOST/` holds it to one machine's files; it runs after the host relabels, so name the machine as it is by then |
| `--crop HOST=NEW:X,Y,W,H` | cut a region of a host's videos out as the videos of NEW, for a recording that is a mosaic of several cameras |
| `--crop-fps` | the frame rate of the cropped videos; default the source's |
| `--flip HOST` | turn a host's videos by 180 degrees |
| `--to-raw HOST` | keep a host's files under `raw/`, out of the sources |
| `--delete-host HOST` | delete a host's files |
| `--tag-size` | the AprilTag size of the session in metres, noted in the session manifest |
| `--note` | a note kept in the session manifest; repeatable |
| `--same-class-as SESSION` | marks another session whose participants come from the same class, writing `same_class_as` into both manifests; the link survives rebuilds and follows a renamed session |
| `--pupils 0,1` | the tag ids of the session's pupils, written as `pupils` into the manifest with a dated note, which `mmla ses-fuse` takes as the session's pupils ([Pupils](../../../analytics/window_features.md#pupils)); survives rebuilds; `--pupils none` (or `''`) clears them |
| `--prune-legacy` | keeps the speaker profiles (as `collection/<host>/profiles/`) and `meta.txt`, and deletes `legacy/` and the old analysis folders (frames, logs, measurements, exports), which a replay produces again |
| `--scope`, `--participant`, `--participants-in-order` | the scope and wearer of each microphone ([Microphone scope and wearers](#microphone-scope-and-wearers)) |
| `-a`, `--artifacts` | the artifacts root; default `<cwd>/artifacts` |

`--crop` and `--flip` re-encode with the Mac's hardware encoder; the original of a flip stays under `raw/`.

### Microphone scope and wearers

Every audio recording in the manifests has a `scope`, `personal` (a microphone worn by one person) or `group` (a room microphone), and a `participant`, the tag id of its wearer or null. The ASR replay and the fusion read them ([Personal microphones](../../../pipelines/asr/speakers-and-diarization.md#personal-microphones-and-energy-attribution)).

| Option | What it does |
|---|---|
| `--scope [HOST/]DEVICE=SCOPE` | sets the scope by hand, `personal` or `group` |
| `--participant [HOST/]DEVICE=TAG` | binds a wearer and makes the microphone personal; `none` unbinds it |
| `--participants-in-order` | tags the personal microphones in natural device order with the tag ids of the session group's participants in `config/experiments.yaml`, lowest first |

```bash
mmla ses-tidy <session-id> --device ch0=vimo-0-ch0 --device ch1=vimo-0-ch1 --participants-in-order --participant vimo-0-ch1=3
```

The device name gives the default scope: `jabra` and a camera's audio track are the group's, `vimo` and `badge` (also an imported `base-vimo-*` folder) a person's, and `mic`, `mix` and `chN` stay unknown until the device is renamed. A live Collection recording already notes the Participant picked on the card; what it left unbound is bound here.

??? info "Details: scope rules"
    - A device rename works its default scope out again (an explicit `--scope` stays) and keeps its tag. The values survive later rebuilds, and a rebuild keeps what a live recording noted.
    - A group microphone never keeps a tag.

??? info "Details: --participants-in-order"
    - Natural device order is `vimo-0-ch0 < vimo-0-ch1 < vimo-1 < vimo-10`. The group is the session's, or the one after `-e` and `-g` when given.
    - A microphone already bound to one of those tags (on the Collection card, or by an earlier `--participant`) keeps it, and the others take the tags left. Any microphone beyond the list is left unbound, and every binding it changes is printed.
    - For a group the file does not list, it falls back to 0, 1, 2 ... and says so.
    - A `--participant` given with it wins.

## Align recordings

`mmla ses-align` puts a session's recordings on one clock and one start. A per-person microphone (badge, vimo) is stamped by the base that recorded it; a camera's file is stamped by whatever wrote it, to the second at best and after the delay of a stream. Both heard the same room, so the command cross-correlates every audio recording with the longest per-person microphone and reports how far each one's nominal start is off.

```bash
mmla ses-align <session-id>                     # measure only
mmla ses-align <session-id> --apply --trim
mmla ses-align <session-id> --end 2400 --dry-run
```

| Option | Default | What it does |
|---|---|---|
| `--reference HOST` | the longest per-person microphone | the recording the others are aligned to |
| `--apply` | off | moves the recordings measured off by more than `--tolerance` with a confidence above `--min-confidence` |
| `--tolerance` | `0.25` | seconds of offset left alone |
| `--min-confidence` | `8` | how far the correlation peak must stand above the rest |
| `--trim` | off | cuts every recording to the session's common start |
| `--end SECONDS` | | ends every recording SECONDS after the common start |
| `--dry-run` | off | with `--end` or an edit: say what would change, change nothing |
| `--window` | `1200` | seconds of audio compared |
| `--max-lag` | `120` | the largest offset looked for, in seconds |
| `-a`, `--artifacts` | `<cwd>/artifacts` | the artifacts root |

Each measurement comes with a confidence, how far the correlation peak stands above the rest (below 8 the number is a guess), and a margin over the runner-up. Two recordings of one base come out at 0.00, which checks that the method works.

??? info "Details: what --apply moves"
    - A camera's video moves with its audio track, since they share one start: every video that starts with the track to the millisecond, on any machine. That covers a track filed under one machine (`<host>/jabra-0`) with its video under another, and the crops of one recording, as `ses-tidy --crop` gives a mosaic's track and its cameras one start. A track never moves without its picture.
    - Nothing else moves along. The output names every file that starts with a moved one and stays: the reference, a per-person microphone, and another audio recording, which each go by their own measurement.

??? info "Details: --trim and --end"
    - `--trim` cuts every recording to the common start, the latest start among them: audio to the sample, video at the last keyframe before it (a stream copy, nothing re-encoded, so a video that starts before a microphone ends up at most one keyframe interval ahead of it). It names every file with its exact start, and what came before the common start is gone.
    - `--end SECONDS` cuts every recording of every host, audio and video, SECONDS after the common start (the manifest's `initial_sync_time`). Audio is cut to the sample; video by stream copy, which stops in decode order, so a video may keep a frame or two past the cut, and the manifest takes the length `ffprobe` reads.
    - A recording that already ends by then, or within 0.25 s after, is left as is, so a second run cuts nothing.
    - Unlike `--trim`, `--end` can be undone: the full-length originals move to `raw/<host>/<audio|video>/`, and the manifests are rebuilt with the new lengths, `stopped_at` and a note naming the cut. `--end` alone measures nothing.

!!! note
    The session's analysis outputs still cover the old length after `--trim`, `--end` or an edit. Replay and fuse the session again afterwards.

### Edit audio files

What no shift of a start can fix is edited inside the audio files named by `--devices` (`vimo-0-ch1`, `<host>/vimo-0`, several separated by commas). The edits run without `--apply`, `--trim` and `--end`.

| Option | What it does |
|---|---|
| `--devices [HOST/]DEVICE[,...]` | the files to edit |
| `--warp MAP` | puts a file on the reference timeline with a piecewise time map, for two channels that lost audio on their own |
| `--cut-zeros TIME[:LENGTH]` | cuts the digital-zero run that starts within 0.5 s of TIME in each file, all of it or its first LENGTH, for the silence an import put in for a clock step; refuses a file that has none there |
| `--cut START:END`, `--cut START+LENGTH` | cuts a stretch |
| `--cut-head SECONDS` | cuts a file's first SECONDS and keeps its start, so the rest sits that much earlier, for a recording measured a constant time late (`--trim` is the head cut that keeps the timing) |
| `--samples` | positions are sample indices, not seconds into the file as it is |
| `--note TEXT` | said in the manifest note of the edit |

```bash
mmla ses-align <session-id> --devices vimo-0,vimo-1,vimo-2 --cut-zeros <seconds> --cut-zeros <seconds> --dry-run
mmla ses-align <session-id> --devices vimo-0,vimo-1 --cut-head <seconds> --note "measured <seconds> s late against jabra-0"
mmla ses-align <session-id> --devices vimo-0-ch0 --warp <map-ch0>.json --dry-run
```

The original moves under `raw/<host>/audio/` with a `<name>.edits.json` of every edit beside it, and the start stamp in the name stays.

??? info "Details: how an edit is written"
    - Nothing is resampled, which would change the pitch of speech: where the lag grows, silence is inserted, and where it shrinks, the samples that would land on audio already placed are cut.
    - A file edited twice keeps its first original under `raw/`.
    - The manifests get the new length and a note with the numbers: the ranges cut, the knots (a summary when a map has more than 12; every knot stays in the `.edits.json`) and `--note`.

??? info "Details: the warp map"
    The map is a JSON file or inline JSON, with an optional `"note"` on how it was measured:

    - `{"lags": [[time, lag], ...]}`: from `time` on, the file's content sits `lag` seconds later; the first lag holds from the start.
    - `{"knots": [[t_in, t_out], ...]}`: the stretch from `t_in` to the next knot starts at `t_out`.

    For a lag track measured as "the other file hears each sound `lag` s later at time `t`", the late file goes onto the early one's timeline with knots `[t + lag, t]`, and the early file onto the late one's with `[t, t + lag]`. Either cuts from the moved file what it heard while the other one dropped audio. Two captures that both dropped some therefore go onto one insert-only timeline instead, one map per file whose lag never falls, so each drop of either capture becomes silence in that capture.
