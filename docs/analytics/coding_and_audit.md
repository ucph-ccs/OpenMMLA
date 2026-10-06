# Coding campaigns and the sensing audit

`mmla ses-code` serves the coding page to one coder at a time on the machine that holds the recordings. For codes that must be blind, such as a second coder, an independent reference sample or an adjudication, it also runs **locked coding campaigns**. A campaign server binds every request to one named coder, shows that coder only their own labels, logs every request in a hash chain, and keeps the labels apart until the campaign is closed and released. It also runs a **sensing audit**, in which a person checks what the camera and microphone pipelines said about a sample of frames and windows without being shown what they said.

The code is in `openmmla/commands/ses/`:

- `code_locked.py`: the campaign server, the links and the request log;
- `code_assign.py`: the window sample;
- `audit.py`: the audit's sample and frozen outputs;
- `audit_render.py`: the audit's images and clips;
- `audit_page.py`: the audit's page;
- `audit_score.py`: the audit's scores.

Without any of the flags below, `mmla ses-code` serves the default page exactly as before: the page's bytes, its routes and its answers are unchanged, and a test checks the page's sha256.

## Network isolation

The tool limits who may connect to its own port. It cannot stop a coder's machine from reaching the server's other ports, and those show what a blind coder must not see:

- the default coding page (port 8765) shows every coder's labels and the agreement page;
- the dashboard shows a model's interaction label per window, the raw recordings and `window_features.csv`;
- the streaming server serves the cameras.

The operator sets this up before the first link and records it in the log:

1. **One port only.** A coder's or auditor's machine must reach exactly one port of the server: 8766, the campaign's or the audit's. Use a Tailscale ACL that lets the coding machines' tag reach only `uber-server:8766`, or an SSH account limited by `permitopen` to `127.0.0.1:8766` (then bind 127.0.0.1).
2. **A probe from each machine.** Run a port probe from each coding machine (for example `nc -zv uber-server 1-65535`) and record what it reached with `--log-note "port probe from <machine>: only 8766 open"`.
3. **Kiosk machines.** Coders use the machines whose addresses `--allow-from` names, never their own devices.
4. **Approvals.** Record the data protection approval of the processing, each coder's and auditor's signed data agreement, and the non-developer who confirmed the list of coders, each with `--log-note`.

The servers enforce what they can:

- **Bind.** `--locked` and `--audit` refuse `0.0.0.0` and LAN or public addresses, since plain HTTP would carry clips, frames and cookies across them. They bind the tailnet address or 127.0.0.1, the latter behind an SSH forward. An open audit (`--audit-open`, [below](#the-open-audit)) binds the same addresses.
- **Allowed clients.** With the tailnet address, `--allow-from` is required and names hosts (single addresses). A network needs `--allow-wide`, which the start line records. Any other client gets 403 and a log line. With 127.0.0.1, only this machine may connect, unless `--allow-from` narrows it further. An open audit is the exception: there `--allow-from` is optional, and without it every tailnet address may connect, as anyone who reaches the default page may. The start line records the addresses allowed.
- **Another instance.** Both servers refuse to start while another `ses-code` serves the same artifacts, unless `--i-know-another-instance-runs`, which the start line records. This check is a convenience: the control is the one-port rule above. The running default page never needs to stop.
- **Their own checkout.** Run the campaign and audit servers from a separate checkout on port 8766 (the default for `--locked` and `--audit`). A campaign and an audit served at once need `-p` for one of them.

## Locked coding campaigns

### What a coder never sees

On a campaign server a coder sees:

- the coding page, under the name their link was issued to;
- the sessions of the campaign, named `R01`, `R02` and so on, in an order of their own;
- the windows assigned to them, with times as minutes and seconds into the recording;
- their own codes, notes and progress;
- the transcript of a window, when the campaign was made with one.

They never see:

- any other coder's labels, notes or counts, nor a model's or the adjudicated labels;
- the agreement page or its API: those paths answer as an unknown path does;
- session ids, dates, groups, splits, the inclusion rule's verdict, devices, file paths or absolute times, in any answer, error answers included;
- how the windows were drawn (`design.json` is never read by the server).

A coder's name, a session and a time sent by the page are never trusted. The server takes the name from the coder's cookie, and it accepts a save only for an assigned window of an alias of the campaign.

### Steps

All commands take `--campaign DIR`. A good place for the folder is `artifacts/runtime/coding/<name>/`. The folder's name (letters, digits, `-` and `_`) names the labels folder.

| Step | Command | What it does |
|---|---|---|
| 1 | `--campaign-init --campaign-sessions ID,ID,…` | Writes `campaign.yml`: the sessions and their aliases (a seeded shuffle, `--assign-seed`), the window and step (`-w`, `-st`), the Transcript button (off with `--campaign-no-text`), the viewing rule (`--min-view full\|none`) and how many preceding windows the `p` key plays (`--prior-windows`, default 1). The inclusion rule and `--hold` do not apply: the ids are exact. |
| 2 | `--draw-assignment N\|all --strata-coders A,B,C` | Optional. Draws the windows to code (below). Without it, every window of every session is coded. Refused once a link has been used. |
| 3 | `--log-note TEXT` | The port probes and approvals of [Network isolation](#network-isolation). |
| 4 | `--issue-token NAME [--link-base http://HOST:8766]` | Writes NAME's link to `links/NAME.txt`, readable by its owner only. The link is never printed. Hand it over in person (on paper or typed), then delete the file. |
| 5 | `--prepare-clips` | Optional. Cuts every assigned window's clip and its preceding windows' clips into the campaign's cache, one at a time with two ffmpeg threads. |
| 6 | `--locked --bind <tailnet address> --allow-from <coder machines>` | Serves the campaign until Ctrl-C. A closed campaign is refused. |
| 7 | `--anchor` | Prints only hashes: the log's head and each coder file's sha256, keyed by alias and link id, never by name. Send them to someone outside the project (the supervisor or the data protection officer) at the start, at the end of each coding day and at the close. |
| 8 | `--close-campaign` | Verifies the log first (below). Then every later save is refused, the clip cache is deleted, and a server still running answers 410 for clips and transcripts. |
| 9 | `--release-campaign` | After the close only. Verifies the log again, then copies each coder's labels to the default `labels/` folder, refusing to overwrite any file. `release.json` lists the copies with their sha256, the log's head and what the check found. Only now do the labels reach `mmla ses-classify`, `labels.load_labels` and the default agreement page. |

Other commands:

- `--revoke-token NAME` takes a link back: its browser is refused at its next request. A lost link or browser is replaced by a revoke and a new `--issue-token`. Both are logged, however they are called.
- `--verify-log PATH` checks a log (below).
- `--despite-log-failure` lets close, release or an audit's score go on when the log does not verify. The step's log line and outputs record the override.

A coder name is refused in these cases:

- it differs from its file-name form;
- it differs only in case from another coder of the campaign;
- it is `adjudicated`;
- a labels file of the default page (a coder's or a model's) of that name exists in a campaign session, in any case.

Who a coder is cannot be checked by the tool. Keeping a person away from a campaign is the operator's job, and the log shows which link coded.

### Links

A link `/c/<token>` works once, in one browser:

- **Opening it** shows a Start button, so a link preview or a browser prefetch never spends it.
- **Pressing Start** claims it. The server stores the sha256 of a new random device secret and sets that secret as an HttpOnly cookie for 30 days. The token itself is stored only as its sha256.
- **A second claim**, from anywhere, shows "this link was already used" and is logged. Coders are told to report that page.
- **The browser that claimed a link** may open it again.

The claim's log line names the device: the first 16 hex digits of the sha256 of the device secret. Every later request of that link names the same device.

### The page

The page is the default coding page with these changes:

- **The coder.** The coder name field is gone; the header says "Coding as NAME". Neither the link's `?coder=` nor the browser's storage changes the name.
- **Viewing.** With `--min-view full` (the default), a class key counts once 95 % of the window's own clip has played. Seeking to the end does not count, and the playback speed is held at 1.25 at most. The server checks the reported share again. A window already coded takes a new code, a note or an also-state without being watched again.
- **Context.** `p` plays the preceding window (or windows, up to `--prior-windows`), then the window again.
- **The viewing record.** Each class key's record carries how the window was watched: the seconds of the clip played, whether it played to its end, the fastest playback rate, the replays and the preceding windows played.

The viewing gate and the viewing record rest on what the page reports of its playback. They keep a coder from coding a window unseen by accident; they are a convenience, not a control, and a paper should describe them so. They describe the new coders only: how coders A, B and C viewed windows is a separate disclosure.

### Files

```
<campaign>/
  campaign.yml       settings, sessions and aliases, coders with the sha256 of their tokens and devices
  assignment.json    the windows each coder codes (served)
  design.json        how they were drawn (never served)
  requests.jsonl     the request log
  links/NAME.txt     a coder's link, until handed over
  clips/             the clip cache, without the recordings' container metadata
  release.json       what the release copied, with sha256s and the log's head
artifacts/<session>/labels/locked/<campaign>/<coder>.jsonl   the coders' labels until the release
```

A label line carries what the page sent, checked and cut to the fields of the codebook and the viewing record. It also carries what the server set: the session id, the absolute window start and end, the coder, the campaign, the link's id, the request's sequence number in the log and the time it was saved.

### The window sample

`--draw-assignment` reads the default-page labels of the coders named in `--strata-coders`, as they stand now, and refuses a file a model or a script wrote.

- **Population.** The windows coded by at least two of those coders, or by the one when one is named. With `--assign-filter non-unanimous`, only the windows they coded differently.
- **Strata.** Any of `lesson` (the session id without its start suffix), `majority` (the code given by more than half of the coders who coded the window, else none) and `unanimity` (`--strata`, default `lesson,majority`).
- **Allocation.** N windows in proportion to stratum size, at least `--min-per-stratum` per stratum (or the whole stratum when smaller), by largest remainder to exactly N. `all` takes the whole population.
- **Draw.** A seeded simple random sample within each stratum.

`design.json` keeps each stratum's population, sample and weight (population over sample), the coders, the labels files with their sha256 and the time they were read. Estimates over the sample (an agreement, a model's κ) are weighted with these weights.

### The request log

`requests.jsonl` has one line per request and per operator action: init, link issue and revoke, assignment, note, anchor, close and release.

- **A request line** holds the time, the coder, the link's id and device, the client's address and browser, the method and path (a link as `/c/*`, never its token), the session as its real id and alias, the window start, the status, the bytes sent and the time taken. Cookies, tokens and device secrets are never logged.
- **A save** carries the sha256 of the exact label line it appended and the file's path. The label line and its log line are both synced to disk before the answer goes back.
- **The start line** records the flags, the software, the bind and the allowed addresses, `--allow-wide`, another instance, and the sha256 of `campaign.yml`. It also records the sha256 of each module the server runs as it was imported, whether the file changed since, its git state and whether the checkout has changed files. A stop line ends each run.
- **`campaign-changed`** is logged with the file's new sha256 whenever a running server finds `campaign.yml` changed: by a claim, by the operator's links and revokes, or by hand.

Every line carries `seq` and `prev`, the sha256 of the line before it. Every writer (the server's threads, the operator's commands, a restarted server) holds a file lock while it reads the last line and appends, so the chain continues across processes and restarts.

`--verify-log PATH` checks:

1. **The chain.** Every line chains to the one before it. A line a crash cut short passes when the next line chains to it (the next writer ends it and chains over it), or when it is the last line and does not end; it is reported apart, and the command exits 3. Any other line that is not JSON is an edit.
2. **The links.** A link is claimed once, and only after a line of the log issued it. Every request of a link comes from the device its logged claim set up, and no request follows the link's logged revoke. So an operator who edits `campaign.yml` to act as a coder (a device hash of their own, a claim or a revoke undone, a link written in by hand) shows in the log.
3. **The saves, both ways.** When `campaign.yml` sits next to the log, every line of the campaign's labels files was saved by a logged request of the same link, in the logged order, and every logged save is in its file.

It prints "intact" with the number of lines and the head, or the first line that fails, and exits 0 (intact), 3 (intact apart from crash cuts) or 1. `--close-campaign`, `--release-campaign` and `--audit-score` run the same check first and refuse when it fails, unless `--despite-log-failure`. A label line whose log line a crash cut away fails the check: nothing tells it from a line added by hand.

**What the log can show.** With the anchors sent outside, the log shows:

- that no line before an anchored head was edited, removed or inserted;
- that the labels files hold exactly the saves the log records;
- which link, device, address and browser made each request;
- that no link was used from a device its claim did not set up.

**What it cannot show.**

- Who sat at the browser a link was claimed in.
- Anything after the last anchor. The operator owns `requests.jsonl` and the labels files (they are the operator's files, mode 0600; no root is needed), so the operator can rewrite the log, the labels and `campaign.yml` together after the last anchor and before the next. The anchors bound that window.
- Whether the operator used a coder's link before the coder did. The coder would then see "this link was already used" and must report it.

`sudo chattr +a requests.jsonl` makes the file append-only at the file-system level, and a person with root can lift it.

### For another server

The sensing audit's server reuses the guard of the campaign server. `GuardedHandler` (a mixin placed before `BaseHTTPRequestHandler`):

- checks the client's address;
- serves the one-time links;
- binds every other request to a claimed link by its cookie;
- checks POSTs (JSON, from the page's own origin, at most 64 KiB);
- answers with `no-store`;
- writes one log line per request.

A subclass lists its routes in `ROUTES` and `PREFIX_ROUTES` and sets `KIND`, the scope a link must hold (`--token-scope code|audit`). `append_record` appends a record and its log line together, under the log's lock, after an optional check that sees every record saved before. So two saves at once are judged one after the other, and `verify_log` can check the record files both ways.

## The sensing audit

### What it checks

An auditor answers four kinds of question, in this order for each recording:

| Question | Unit | What the auditor sees | Answers |
|---|---|---|---|
| Roster | a crop of a person whose badge the camera read | the person in a yellow box, a diamond on the badge | the badge is theirs: yes, no, cannot tell |
| Identity | one camera's frame | every person box the pose model drew, numbered left to right, and the confirmed roster crops of each pupil (A, B, C) | per box: pupil A, B or C, someone not in the group, not a person, cannot tell; and how many group members the frame shows without a box |
| Gaze | one box of a frame | that box highlighted and a close-up of the head | where the person looks, in the order the pipeline breaks ties (below), between two of these (both named), or cannot tell |
| Who speaks | one 10 s window of a recording with a group microphone | every camera in a grid with the group microphone | no one, a group member, the teacher or another adult, another group, cannot tell |

The roster crops a person confirms become that person's reference pictures of the pupils in the recording. The pupils are named A, B and C, never by badge.

The identity answers of a frame are locked once saved, and only then does the gaze question open. It asks about every box the auditor gave a pupil and every box a frozen pipeline version calls a pupil. The head close-up is cut around the head box the pose model's own keypoints give (`features.head_box`), not around the stored face box, which sits on the wrong head whenever face assignment failed.

The gaze classes follow the pipeline's tie-breaking (`features.gaze_target`, then the work area). Take the first that applies:

1. a group member's face, or 2. the face of someone not in the group: any face comes before any hands;
3. their own hands, 4. a group member's hands, or 5. someone else's hands, with what they hold: among hands, whoever's hands the gaze lands nearest;
6. the task material or work area on the table, in nobody's hands;
7. somewhere else in the picture;
8. outside the picture.

`9` names two of these when the gaze lands between them, and `x` is cannot tell.

The audit has no presence question, no speech-activity or overlap question, no word error rate and no position check.

### What an auditor never sees

- **No pipeline conclusion.** In the default blind mode, nothing the pipeline concluded is drawn or sent: no badge ids, no tag sources, no gaze classes and no speech measures. The boxes are the pose model's own, and every person gets one.
- **One fact, after the lock.** The page learns which boxes a frozen version calls pupils, and only after the frame's identity answers are locked.
- **No pipeline file.** The server reads the plan, the views and the answers files. It never opens a `pipeline*.json` file (a test checks this).
- **Verify mode.** With `--audit-mode verify` at sampling, the renderer draws the display version's pupil and tag source beside the member boxes, and its gaze ray and class on the gaze pictures. A sample is in one mode for good.
- **No identifying detail.** Session ids, dates, devices, file paths and absolute times never reach the page: recordings are `R01`, `R02` and so on, and cameras are numbered.
- **No other auditor.** Every auditor comes in through a one-time link of scope `audit`. No answer is saved, and nothing is shown, under a name the page gives, so no auditor reads or answers as another. In an open audit the auditor is the name typed: the server shows each name only its own answers and progress, but it cannot tell who typed a name ([below](#the-open-audit)).

### Versions

The answers do not depend on the pipeline version. Frames are drawn by their times alone, and windows the same way. Each version's outputs are frozen apart, in `pipeline_reported.json` and `pipeline_rerun.json`. Each version's persons are matched to the displayed boxes by their overlap (IoU 0.5 or more). One pass of answers therefore scores both the version a paper reports (`reported`: the outputs from before the 2026-10-02 re-run, with the old fused tables of `~/fusion_backup_20261002` and the old `vfa_features` events from the backups) and the re-run (`rerun`).

The display version (`--audit-version` at sampling) gives the boxes. Its stored frames are those the bases' boxes come from, and person boxes come from the same pose model on the same frames in every version.

A version is frozen from its own `vfa_features` events and its own fused tables. Before anything is drawn or frozen, each session must pass these checks:

- the table's sha256 is the one its `parameters.json` recorded;
- the count of the events is the count the fusion read;
- the fused view reproduces the table. The audit replays `window_features`' own frame steps: the tag memory, the face's refusals of remembered tags when the table took them, the hand circle, the work areas, the tags carried along the tracks and the pupils' seats. Every kept person's camera counts and gaze shares per window must equal the table's.

A session that fails one of these is left out with the reason, unless `--audit-allow-drift`, which records the problems. Freeze the version a paper reports before its events are overwritten by a re-run.

### Which frame

Two checks make sure a frame set is the video frame the bases read:

- **The sync time.** Every frame set's time must be, within 1 ms, the time a base of the replay config acquired its keyframe: the sync time, the keyframe interval added k times to the file's own offset, plus the file's start (`vfa_base._process_keyframes`). Otherwise the session is left out, whatever `--audit-allow-drift` says. A sync time the TUI rewrote, or another interval, would put every item on another frame. A config that sets no sync time takes each base's own folder, as the bases do, and refuses bases that would start from different times.
- **The same frames in every version.** A second version's replay config must have the design's sync time, interval and video files, and its frame sets must sit at the items' times. Otherwise it is refused for that session. A refused freeze can be tried again once the version's files are right.

### Steps

| Step | Command | What it does |
|---|---|---|
| 1 | `--audit-sample ID --audit-version reported --audit-events DIR --audit-tables PATTERN` | Chooses the sessions, draws the items, writes the plan, the views and the designs, and freezes the display version. The sessions are those of `--audit-sessions ID,...`, else every session with video, a fused table and a replay config, less `--hold` unless `--audit-include-held`. `PATTERN` holds `{sid}`. `DIR` holds `<sid>.json.gz`, `<sid>.json` or an export `<sid>/`; without it the events come from InfluxDB (`--influx-config`). It also writes `campaign.yml` in the audit's folder, for the links. |
| 2 | `--audit-freeze ID --audit-version rerun --audit-events DIR --audit-tables PATTERN` | Freezes another version's outputs for the same items. Freeze both versions before rendering, so the gaze question asks about the member boxes of both. |
| 3 | `--audit-render ID` | Decodes, checks and draws the frames, crops the heads and the roster, and cuts the clips (below), in `vfa-base`. |
| 4 | `--campaign artifacts/runtime/audit/ID --log-note TEXT` | The port probes and approvals of [Network isolation](#network-isolation). |
| 5 | `--campaign artifacts/runtime/audit/ID --issue-token NAME --token-scope audit` | A one-time link for an auditor, written to `links/NAME.txt` and never printed. `--token-subset reliability` limits a second auditor to the reliability subset. An open audit skips this step. |
| 6 | `--audit-estimate ID` | The hours the answers take, from assumed seconds per judgement, or from the practice answers once there are some. |
| 7 | `--audit ID --bind <tailnet address> --allow-from <auditor machines>` | Serves the page until Ctrl-C. It refuses when no audit link is issued, and once the audit is closed. With `--bind 127.0.0.1` it is reached through an SSH forward. With `--audit-open`, no link is needed and `--allow-from` is optional ([The open audit](#the-open-audit)). |
| 8 | `--campaign artifacts/runtime/audit/ID --anchor` | Prints only hashes: the log's head, `campaign.yml`, the plan, the render index, per recording its design, view and each frozen version, and each answers file. Send them outside the project before the first answer, at the end of each day and at the close. |
| 9 | `--audit-score ID --audit-version reported`, then `rerun` | Scores a version against the answers (below). |
| 10 | `--campaign artifacts/runtime/audit/ID --close-campaign`, then `--audit-purge ID` | Closes the audit to answers (a server still running answers 410 for pictures and clips), then deletes every image and clip. The plan, views, frozen outputs, answers and scores are kept. |

Every step logs the sha256 of the files it wrote in the audit's request log. Two rules keep the frozen outputs fixed:

- **A version is frozen once.** A second freeze of a version a log line already froze is refused, also when its file was deleted.
- **Not after the answers.** Freezing and rendering refuse once a scored (non-practice) answer exists, unless `--audit-after-answers`. The step's log line, the frozen file and every score's header then name it.

A freeze made after the render prints a note: the gaze question asks only about the boxes the versions frozen at render time call pupils. Render again before any answer to ask the new ones; the scores count the others as member boxes not asked.

`--log-note`, `--revoke-token` and `--verify-log` work on the audit's folder as on a campaign's, and `--verify-log` checks the answers files both ways. The campaign commands that draw, serve, cut clips for or release a coding campaign refuse an audit's folder. An audit id is letters, digits, `-` and `_`, in every step.

### The sample

All draws are seeded (`--audit-seed`) and use frame times only, never a pipeline output. A frame is a second at which the display version holds a frame of that camera, more than `--audit-margin` seconds (10) from either end of the session. A frame of a camera that shares its angle with others, in a frame set that lost or gained a frame, is not used: its place in the set no longer names its camera.

| Part | Default | Flag | How it is drawn |
|---|---|---|---|
| Identity and gaze frames | 50 member judgements per lesson | `--audit-person-frames` | 50 divided by the group size frames per lesson (25 for a pair, 17 for three). They are shared across the lesson's sessions in proportion to their frames and evenly across cameras, and drawn evenly spaced in time from a random start. A frame's weight is its camera's frames over the frames drawn from it. |
| Roster crops | 4 per pupil | `--audit-roster` | Frames whose stored person read the pupil's badge (torso or box) with the badge inside the box, never an identity frame, spread over cameras and time. |
| Who-speaks windows | 10 per lesson with a group microphone | `--audit-speech` | The fused table's windows inside the group microphone's and the cameras' recordings, a simple random sample per lesson. |
| Reliability subset | 20 % | `--audit-reliability` | A share of each lesson's frames and windows, flagged for a second auditor. |
| Practice | 12 | `--audit-practice` | From the first DEV lesson in the order: up to 4 who-speaks windows the draw left, and frames for the rest. They come first and are never scored. |

With the defaults over sixteen lessons, the primary auditor needs about 3 to 4 hours (`--audit-estimate`). The second auditor needs the reliability subset, the rosters of its recordings and the practice. Plan sittings of at most 90 minutes.

The order of the recordings is a seeded shuffle, the same for every auditor except the reliability subset, which runs in reverse. Within a recording, the page asks the roster first, then the frames, then the windows.

### The frame check

`--audit-render` decodes each frame the way the base did: from the sync time, it adds the keyframe interval k times to the file's offset and reads through `VideoStream` in file mode, with the rotation and fisheye remap the config sets. The decoded size must equal the stored frame's.

The renderer then finds the AprilTags on every decoded frame that has stored tags. It uses `cv2.aruco` with the 36h11 family and corner refinement, and finds tags whose outline is at least 1 % of the frame's larger side. A tag's centre is where its diagonals cross, as `pupil_apriltags` reports it. A frame matches when every tag it finds again lies within 2 px of where it was stored (`--audit-tag-px`).

- **An item whose own frame does not match** is a render error, and its pictures are deleted, whatever its camera's rate.
- **A camera's match rate** is its matching frames over its frames with any tag found again, including some extra frames per camera (`--audit-check-frames`). A camera under 95 % (`--audit-min-tag-match`), or with fewer than 3 checked frames (`--audit-min-tag-frames`), is not served: its items are render errors, and its images are deleted.

The bounds are set at sampling and recorded in the plan. A flag given at rendering overrides them, and the change is recorded.

The check catches a seek that lands on another frame, a capture with a variable frame rate and a wrong video file. `render_index.json` keeps:

- per camera, the rates and the median offset of the tags found again;
- per item, its check and each head's brightness;
- the errors;
- per recording, whether its replay config is still the file the sampling checked. The render uses the design's sampled values either way.

A frame or a clip that fails is its item's error, and a recording that fails is reported, while the others go on.

### The page and keys

| Part | Keys |
|---|---|
| Roster | `y` yes · `n` no · `x` cannot tell. Each key saves and moves on. The roster of a recording can change until its first frame is answered. |
| Identity | `a` `b` `c` pupil · `o` not in the group · `p` not a person · `x` cannot tell, for the selected box (click a box or press `⌫` to go back) · then `0` to `3` members without a box · `f` flags a broken frame · `Enter` saves and locks |
| Gaze | `1` to `8` where (above) · `9` between two, then the two · `x` cannot tell · `⌫` back · `Enter` saves |
| Who speaks | `0` no one · `1` a group member · `2` the teacher or another adult · `3` another group · `x` cannot tell · `space` replays. The clip must play to its end first. |

Everywhere but the roster, `n` focuses the note. `←` and `→` move between items, and the page starts at the first unanswered item. Notes are kept with the answers and never read into the scores. The who-speaks gate (the clip played first) rests on what the browser reports: a convenience, not a control.

### The open audit

`--audit ID --audit-open` serves the audit the way the default coding page (port 8765) is served: no link, claim or cookie. Each auditor types their name and chooses what they answer. Every other rule of the audit stays.

**Operator steps.** Sample, freeze and render as above. Skip `--issue-token`. Then:

1. `--audit ID --audit-open --bind <tailnet address>` serves the page; add `--allow-from <auditor machines>` to narrow it. `--bind 127.0.0.1` behind an SSH forward works as before. `0.0.0.0` and LAN addresses are still refused.
2. Tell each auditor the name to type, spelled the same way at every sitting, and whether they do the full audit or the reliability subset.
3. Anchor, score, close and purge as above.

The open server serves no link: `/c/<token>` answers 404, and a link's cookie names no one.

**The page.** It opens on a form:

- **Name.** Trimmed, not empty, and at most 100 bytes in its file-name form, as the default page checks a coder name. It may not hold a control, formatting or line-break character, since such a name could forge or hide lines of the scores' report. A name typed in two Unicode forms (an accent typed as one character or as two) is one name. A coder name of the default page may be typed: audit answers go to a file of their own.
- **What you answer.** The full audit, or the reliability subset of the second auditor.

The browser remembers both and fills them in at the next visit. On a shared machine, the next person must change the name. "change name" in the header brings the form back.

**Names.** A name is taken as typed (trimmed, in one Unicode form), and no two names share answers. The server refuses a name in three cases:

- **Another name's file.** The name's answers file, in any case or Unicode form, already holds another name's answers. For example, `Ana` when `ana` has answers, or `ana_b` when `ana b` has answers, since both write `ana_b.jsonl`. The refusal does not say the other name, since typing it would show that auditor's answers.
- **A link's file.** The name's answers file holds answers saved through an audit link.
- **Another scope.** The name's answers so far were saved under the other scope.

The server checks all three again under the request log's lock when it saves an answer, so two first saves at once cannot pass both.

**Links and typed names.** Serve an audit one way. An audit with answers saved through links may still be served open, but the names of those auditors are refused, so let them finish through their links first. Once the open page has saved an answer, `--audit ID` without `--audit-open` refuses to serve the audit, since a link's page would show the answers typed under its name as its own.

**Answers.** They go to `answers/<the name's file-name form>.jsonl`, one line per save as above. Each line has:

- the auditor's name as typed;
- `open: true`;
- the scope (`full` or `reliability`) and its subset;
- no link (`token_id` is null).

The per-name rules hold as before:

- no pipeline answer is served in blind mode;
- a frame's identity answers are locked before its gaze question opens;
- each name sees only its own answers, progress and reference pictures.

**What the log shows.** The request log stays on, and its start line records `open: true` with the addresses allowed. Every request but the page itself carries the typed name, and its log line keeps it as `coder`, with the scope and the client's address and browser. A request without a usable name is refused (400) and logged without one. Every save carries the sha256 of the line it appended, so `--verify-log` checks the answers files both ways as before. It also checks that each open line names the auditor its request typed. The anchors print each file under a hash of its file name, never the name.

**What the log cannot show.**

- Who typed a name. Names are typed, not authenticated, and no device is bound to a name.
- Whether one person answered under two names, or two people under one.
- Whether a person typed another auditor's name. That person sees the other's answers and progress and can answer as them, by mistake or not.
- Whether identity blindness held for a person. Blindness holds per name. A person can answer a frame's identity under a second name, see in its gaze question which boxes the frozen versions call pupils, then answer that frame's identity under their own name, still unlocked. The answers themselves look blind.

The only trace is the address and browser of each request. The scorer lists every save under another name from an address and browser the primary auditor's saves came from, and among them every identity answer of a frame saved before the primary's own ([Scores](#scores)). The same address and browser are no proof of one person: a shared machine, or an SSH forward that shows every auditor as 127.0.0.1. Different ones are no proof of two.

Use links when these matter, for example for a second auditor whose blindness to the first a paper reports. The scores of an open audit say it was open.

### Answers

`artifacts/<session>/audit/<ID>/answers/<auditor>.jsonl` gets one line per saved answer. Each line holds:

- the item and the phase;
- the session, the auditor and the link's id (in an open audit no link, `open: true` and the scope);
- the mode and the codebook version;
- whether it is practice or reliability;
- the answer and the seconds spent;
- the page's and the server's times;
- the request's sequence number in the log.

The server refuses a value, box or pupil the item does not have. The locks (a frame's identity once saved, a recording's roster once a frame of it is answered) are checked under the request log's lock against the answers as saved by then, so two saves sent at once cannot both pass. Of a phase's lines, the last counts, except for identity, whose first counts.

### Scores

`--audit-score ID --audit-version V [--audit-auditor NAME] [--audit-boot 2000]` writes `summary.json`, `report.txt` and CSV tables to `artifacts/runtime/audit/ID/scores/V_<time>/`. Before scoring, it checks:

1. **The log.** The request log verifies with the answers files both ways, unless `--despite-log-failure`.
2. **The frozen files.** Each file of the version is the one its freeze logged, frozen once. A file changed since, frozen twice or frozen by no logged step is refused, with no override.
3. **The tables.** The version's tables and parameters have not changed since the freeze, unless `--audit-allow-drift`.

What is scored:

- **Only linked answers.** Only the answers saved through a claimed audit link of the audit's `campaign.yml`, for that link's own auditor, are read. The others are counted and left out.
- **The first identity.** An identity line after the first of its frame is counted as a lock broken and left out.
- **The primary auditor** is the one named, else the one auditor whose audit link has no subset. Two such auditors need `--audit-auditor`. The other auditors give the inter-auditor agreement.

An open audit is scored by typed name. The scorer finds it in the request log: a start line that served the audit with `--audit-open`. No flag is needed at scoring.

- **The answers.** Those the open page saved count under the name each was saved under, beside any answers of claimed audit links.
- **The primary auditor** is `--audit-auditor NAME`, else the one name whose answers chose the full audit. Two such names need `--audit-auditor`.
- **The agreement** is the primary auditor's with each name that chose the reliability subset, on the items both answered.
- **Other names** are counted and left out. `report.txt` lists every name with its scope, its number of items and its role.
- **Saves under other names from the primary's browser.** From the request log, `report.txt` and `summary.json` list the saves under another name from an address and browser the primary's own saves came from. They also list every identity answer of a frame among them saved before the primary's identity answer of that frame: from then on, that name's gaze question showed which boxes the versions call pupils. The header gives both counts when there are any, and the scorer prints them.
- **The header** of every output says "the audit was open (names typed, not authenticated)".

An answer marked open counts only when the log shows the audit served open. Otherwise it is left out as an answer without a link.

Left out of every score, and counted:

- practice items;
- render errors, and frames whose own tag check failed;
- flagged frames and windows;
- frames the scored version holds no frame of;
- member boxes of a pupil the display roster gives no letter;
- cannot-tell answers.

Each file is headed by:

- the plan's sha256 and the version;
- the drift check and the auditor, and whether the audit was open;
- the log's check and head;
- any version frozen, or render made, after scored answers or after the render.

| Check | Measures |
|---|---|
| Identity | precision of the version's member boxes (correct over correct, swapped, outside the group and not a person) with the error mix, by tag source (read, server track memory, carried along the track), for untagged boxes at a missing pupil's seat, and for the server's stored naming; recall of the members the auditor saw, the misses split into undetected, untagged and mistagged; missed members per frame |
| Gaze | readability (the version's unknown against the auditor's cannot tell); accuracy, Cohen's κ and confusion over the eight classes and member-directed against not, on the boxes both could read; between answers apart; the same on boxes whose identity is correct; by face source, face height and head brightness (terciles fixed in the plan) |
| Who speaks | speech present against `speech_ratio` > 0, > 0.1 and > 0.3; the share of windows with measured speech whose speaker is not the group's, unweighted and weighted by `speech_ratio`, and at `speech_ratio` ≥ 0.3; where worn microphones ran, member-attributed words against the speaker |

Every table comes per lesson first, with Wilson intervals (item-level, approximate), then by split, camera count and group size, then pooled. All rows after the per-lesson ones have lesson-cluster bootstrap intervals (`metrics.session_bootstrap` over the lessons in the row). Design-weighted values sit beside unweighted ones.

### Files

```
artifacts/runtime/audit/<ID>/
  plan.json           sessions, aliases, lessons, sizes, seed, mode, frame check bounds, declared analyses
  campaign.yml        the auditors' links (code_locked; none in an open audit) and whether the audit is closed
  requests.jsonl      the request log, with every step above and the sha256 of what it wrote
  media/<alias>/      images and clips (--audit-purge deletes them)
  render_index.json   frame checks, tag offsets, head brightness, replay config state, errors
  scores/             the scorer's outputs
artifacts/<session>/audit/<ID>/
  view.json           what the server serves
  pipeline.json       the sampling design, the replay config's sha256 and the display version's stored frames (never served)
  pipeline_<v>.json   a version's outputs per item (never served)
  answers/<auditor>.jsonl
```

Images and clips of children stay on the machine that holds the recordings. Exclude `runtime/audit/*/media/` from any copy of `artifacts/`, and purge it once the scores are final. The answers files, notes included, sit in the session folders, so a copy of a session folder carries them.

## A note on short flags

The campaign and audit flags share prefixes with older flags. Abbreviations that used to select an older flag are now ambiguous, and argparse refuses them: `--pre`, `--prep` and `--pr` (`--prepare-text`), `--t` (`--threads`) and `--i` (`--influx-config`). Spell flags out in scripts. No script in the repository uses these abbreviations.
