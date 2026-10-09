# Coding campaigns and the audits

`mmla ses-code` serves the coding page to one coder at a time on the machine that holds the recordings. For codes that must be blind, such as a second coder, an independent reference sample or an adjudication, it also runs **locked coding campaigns**. A campaign server binds every request to one named coder, shows that coder only their own labels, logs every request in a hash chain, and keeps the labels apart until the campaign is closed and released. It also runs a **sensing audit**, in which a person checks what the camera and microphone pipelines said about a sample of frames and windows without being shown what they said, and a **transcription audit**, in which a listener writes down what is said in sampled windows before being shown the text the system made of them.

The code is in `openmmla/commands/ses/`:

- `code_locked.py`: the campaign server, the links and the request log;
- `code_assign.py`: the window sample;
- `audit.py`: the audit's sample and frozen outputs;
- `audit_render.py`: the audit's images and clips;
- `audit_page.py`: the audit's page;
- `audit_score.py`: the audit's scores;
- `audit_speech.py`: the transcription audit's sample, frozen texts, blind close and references;
- `audit_text.py`: the transcription audit's grammar, normalisers and aligner;
- `audit_transcript_page.py`: the transcription audit's page;
- `audit_score_transcript.py`: the transcription audit's scores.

Without any of the flags below, `mmla ses-code` serves the default coding page at `/code` (port 8765 unless `-p`) and the agreement page at `/agreement`, and the start line prints the page's address. The address alone (`/`) and `/audit` redirect to `/code` with their query, so the agreement page's links (`/?session=...&start=...`) open the coding page at their window. The flags below leave the default page, its routes and its answers as the release has them, and a test checks the page's sha256.

The coding page's header names the coder, "Coding as NAME", with **change name** beside it. **change name** turns the name into a field in its place, and the page's keys are off while the field has the focus. Enter takes the name when the server accepts it, with the checks it applies to every save (at most 100 bytes in its file-name form, not the name of a model's labels), and the browser remembers it; a refused name stays in the field with the reason. Esc or leaving the field keeps the name as it was. While the server checks a name sent with Enter, the field waits for the answer even when it loses the focus, and the page codes nothing: the keys do nothing, and a class button says that the name is still being checked. Only Esc drops the check. On the first visit the field is open for a name, and leaving it takes a name typed there as Enter does.

Enter and space on a button or the session list that has the focus (**change name**, **Transcript**, **Auto-advance**, a class button, a definition's arrow) press or open it, as anywhere in a browser, and are none of the page's keys; any other key still is. A click hands the keys back to the page: the session list keeps the focus a click gives it, and space there still replays. On a phone the page is laid out at the phone's width, with the class buttons below the video.

Two links set the name. `?coder=NAME`, as the agreement page's **adjudicate** links write it, codes under NAME for that visit only: the header adds ", for this visit", the remembered name stays, and a name taken through **change name** ends the visit. `?name=NAME`, as the audit entry writes it, is taken as if typed into the field and remembered, then removed from the address.

`--entry-port PORT` puts **← All tasks** at the left of the header: a link back to the entry of two audits served together ([Serving both audits at one address](#serving-both-audits-at-one-address)). It opens the address alone of PORT on the host the page was opened at, with the same protocol, so give it the audits' port (8766 unless they run with `-p`). The link looks like **change name**. A key pressed while it has the focus reaches none of the page's keys, and Enter follows it. Leaving through it is leaving the page: a note not yet saved is lost, as when the tab is closed.

Without the flag the header has no such link. The flag goes with the default page only: beside a campaign's or an audit's flags it is refused, and a campaign's page never shows the link. A port outside 1 to 65535, or the page's own port, is refused too.

## Network isolation

The tool limits who may connect to its own port. It cannot stop a coder's machine from reaching the server's other ports, and those show what a blind coder must not see:

- the default coding page (port 8765) shows every coder's labels, the agreement page and, with its Transcript button, the ASR text of every window;
- the dashboard shows a model's interaction label per window, the raw recordings and `window_features.csv`;
- the streaming server serves the cameras.

The operator sets this up before the first link and records it in the log:

1. **One port only.** A coder's or auditor's machine must reach exactly one port of the server: 8766, the campaign's or the audit's (both audits', when a sensing audit and a transcription audit are [served together](#serving-both-audits-at-one-address)). Use a Tailscale ACL that lets the coding machines' tag reach only `uber-server:8766`, or an SSH account limited by `permitopen` to `127.0.0.1:8766` (then bind 127.0.0.1). The entry of two audits served together links to the coding page on port 8765; `--audit-code-port 0` leaves that link out, since an auditor's machine cannot reach it.
2. **A probe from each machine.** Run a port probe from each coding machine (for example `nc -zv uber-server 1-65535`) and record what it reached with `--log-note "port probe from <machine>: only 8766 open"`.
3. **Kiosk machines.** Coders use the machines whose addresses `--allow-from` names, never their own devices.
4. **Approvals.** Record the data protection approval of the processing, each coder's and auditor's signed data agreement, and the non-developer who confirmed the list of coders, each with `--log-note`.

The servers enforce what they can:

- **Bind.** `--locked` and `--audit` refuse `0.0.0.0` and LAN or public addresses, since plain HTTP would carry clips, frames and cookies across them. They bind the tailnet address or 127.0.0.1, the latter behind an SSH forward. An open audit (`--audit-open`, [below](#the-open-audit)) binds the same addresses.
- **Allowed clients.** With the tailnet address, `--allow-from` is required and names hosts (single addresses). A network needs `--allow-wide`, which the start line records. Any other client gets 403 and a log line. With 127.0.0.1, only this machine may connect, unless `--allow-from` narrows it further. An open audit is the exception: there `--allow-from` is optional, and without it every tailnet address may connect, as anyone who reaches the default page may. The start line records the addresses allowed.
- **Another instance.** Both servers refuse to start while another `ses-code` serves the same artifacts, unless `--i-know-another-instance-runs`, which the start line records. This check is a convenience: the control is the one-port rule above. The running default page never needs to stop.
- **Their own checkout.** Run the campaign and audit servers from a separate checkout on port 8766 (the default for `--locked` and `--audit`). A sensing audit and a transcription audit share that port when one server serves both (`--audit SENSING --audit-with TRANSCRIPTION`, [below](#serving-both-audits-at-one-address)). A campaign and an audit served at once, or two audits served by separate servers, need `-p` for all but one of them; the one-port rule then names the port each coder or auditor reaches.

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

- **The coder.** The header says "Coding as NAME", with no **change name**: the link binds the name. None of `?coder=`, `?name=` and the browser's storage changes it.
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

The sensing audit has no presence question, no speech-activity or overlap question, no word error rate and no position check. The [transcription audit](#the-transcription-audit) asks whether anyone speaks and whether voices overlap, and gives a word error rate.

### What an auditor never sees

- **No pipeline conclusion.** In the default blind mode, nothing the pipeline concluded is drawn or sent: no badge ids, no tag sources, no gaze classes and no speech measures. The boxes are the pose model's own, and every person gets one.
- **One fact, after the lock.** The page learns which boxes a frozen version calls pupils, and only after the frame's identity answers are locked. The transcription audit has one fact of this kind too: the text the content model read, shown only after the operator closes an auditor's blind pass ([Blindness and the reveal](#blindness-and-the-reveal)).
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

Everywhere but the roster, `n` focuses the note. `←` and `→` move between items, and the page starts at the first unanswered item. Enter and space on a button or link that has the focus, such as **change name** reached with Tab, press or follow it and never save: Enter there does not lock a frame's identity answers. Notes are kept with the answers and never read into the scores. The who-speaks gate (the clip played first) rests on what the browser reports: a convenience, not a control.

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

The browser remembers both and fills them in at the next visit. Opened from the entry of two audits ([Serving both audits at one address](#serving-both-audits-at-one-address)), the page shows no form: it starts at once with the name and scope chosen there, which the entry hands to the page in that browser tab only. The entry's `#start` address typed, bookmarked or opened in another tab shows the form, whatever the browser remembers. A reload, or a step back or forward to the page, goes on under the name and scope its tab last started with, whatever another tab started with since. Any other visit shows the form, in the same tab too: an address typed, a link or a bookmark. A name or scope the server refuses shows the form, with the reason.

The header says "Auditing as NAME", with **change name** beside it. **change name** turns the name into a field in its place, filled in, and the page's keys are off while it is open. Enter starts under the name typed, with the scope in use and the checks of **Start**; a refused name stays in the field with the reason, and the page goes on under the name it had. A name taken drops the item shown, so nothing answered under the name before is saved under the new one. Esc or leaving the field keeps the name. The other scope is chosen on the entry or on the form of a new visit. On a shared machine, the next person must change the name.

The server sees none of this: a start without the form sends the requests a **Start** sends, and the request log holds the same lines.

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

## The transcription audit

A transcription audit checks the text the speech pipeline gives a content model. A listener who knows the recordings' language writes down what is said in sampled 10 s windows and answers the content model's own questions about them, blind to every system output. Only once the operator closes their blind pass does the page show them, window by window, the text the content model reads, which they rate. The scores compare that text with the transcripts, and the content model's scores with what the listener heard.

It is drawn with `--audit-sample ID --audit-task transcript` (the plan's `task` is `transcript`) and then runs through the same flags as the sensing audit, with its own folder, links, request log, frozen versions and scores. Every rule of the sensing audit holds unless this section says otherwise. The listeners' conventions are in the [Transcription guide](transcription_guide.md), in Danish with an English twin.

### What the transcription audit checks

The unit is one 10 s window `[ws, ws + 10)` of the display version's fused grid, the window a content model scores. The content model reads a window's ASR text, with the 10 s before it as context, and gives three letter probabilities: a teacher speaks (`teacher`), pupils talk with each other about the task (`peer_task`) or about something else (`peer_other`). A word belongs to the window in which it begins, as in the text the model reads.

Each window's clip spans `[ws − 10, ws + 12)`, 22 s: the model's 10 s of context, the shaded window and a 2 s tail. It has the sensing audit's camera grid (up to four cameras in 640 × 360 tiles) and the session's sound as AAC-LC at 96 kb/s, mono, 16 kHz, with no filter and no change of loudness. Only the shaded window is transcribed.

| Kind of sound | When | The clips |
|---|---|---|
| `group` | the session has a group microphone | the group microphone |
| `dual` | the session has a group microphone, and its personal microphones' chunks hold at least 5 % of the version's words | the group microphone, and a second clip with the sum of the personal microphones (`amix`, not normalised) |
| `mix` | the session has no group microphone | the mix of the worn microphones that replay wrote, `analysis/group_mix/audio_<device>_<start>.wav`; the sha256 of its note is recorded |

Per window the auditor answers in two phases:

| Phase | When | Answers |
|---|---|---|
| `transcribe` | until the operator closes the auditor's blind pass | whether anyone speaks (`speech`, `unintelligible`, `none`); the transcript, one tagged line per turn; whether voices overlap; who speaks (the sensing audit's question); whether an adult speaks; whether pupils of this group talk with each other, about the task or about something else; whether pupils of another group talk off the task; a flag and a note |
| `reveal` | after the close, for the primary auditor only | whether the content model's text conveys what was said (`yes`, `partly`, `no`, `nothing_said`); whether it holds words nobody said (`none`, `some`, `most`, `cannot_tell`); a flag and a note |

The questions about the adult and the pupils are the content model's own, with one change: the page asks about the pupils of *this* group, whom a listener can tell from another group's pupils and the model cannot.

### Blindness and the reveal

Until the operator closes an auditor's blind pass, no response of the server carries any system output: no text, word count, speech measure, score, stratum, rank, weight or session id. The server reads the plan, the views, the answers files and `blind_closed.json`, and never a `pipeline*.json` file.

The only system output the page ever shows is the **reveal**: the display version's text of the window and of the 10 s before it, as the content model reads them. The render copies it into the view, and the server sends it with an item only when:

- the auditor's name is in `blind_closed.json`;
- the auditor saved a transcription of the item before the close;
- the item is not practice;
- the auditor answers the full audit, not the reliability subset.

The reveal shows the auditor's own transcript, read-only, and the content model's text: the previous 10 s in grey, then the window, its lines joined by newlines with no speaker labels. An empty window says that the system wrote nothing and the content model was not asked. The stratum of an item is never shown.

**The blind close.** `--audit-close-blind ID --audit-auditor NAME` closes one auditor's blind pass:

1. It verifies the request log, unless `--despite-log-failure`.
2. Under the log's lock, which every save of the server also holds, it adds NAME to `artifacts/runtime/audit/ID/blind_closed.json` with the sequence number of its own log line and the time. A save is therefore judged wholly before or wholly after the close.
3. It logs `audit-close-blind` with the file's sha256.

It refuses a closed audit, a name closed already, and a name that saved no transcription, which catches a typo. From then on the page refuses that name's transcriptions (409, "the blind pass is closed") and asks for a reveal rating of each item the name transcribed before the close. Practice items, and items with no transcription before the close, ask nothing more.

The page acts on `blind_closed.json`, but the record of a close is its line in the request log. The server does not start, and a close of another name refuses, while the file names a close the log does not hold, lacks one it holds, or holds one at another sequence number. The scorer and the reference export count by the logged closes. If a close stopped between its two writes, the file holds a close the log does not: close that name again, which logs the close at the new step's own sequence number and records the old one as `unlogged_seq`. Any other difference is a hand edit: put the file back as the log has it.

!!! warning "The declared order of the closes"
    Close the primary auditor only once the reliability auditor has finished, or log a `--log-note` that says why not. Decide whether sweep 2 is served, and log it, before this close. The reliability auditor is never shown a reveal, closed or not.

### The two texts

As in the sensing audit, the answers do not depend on the version, since windows are drawn by their times. Each version's text is frozen apart, in `pipeline_rerun.json` and `pipeline_reported.json`.

- **`rerun`** is the display version and the declared primary: the `asr_transcription` events as they are at the freeze, with each session's current fused table. The sample is drawn on its grid, `--audit-sample` freezes it, and the reveal shows its text.
- **`reported`** is the secondary: the events and the tables that a content model's earlier scores were made from. It is frozen right after the sample with `--audit-freeze`, and only with `--audit-asr-reference` and a content arm whose file has a `cur_sha` column, so that it is always checked against the text the content model read. A window missing from its table's grid keeps its text and is listed in the file's `off_grid`.

Per window, a frozen file holds:

- `consumed`: the text of every source, as the content model reads it (the inside parts of `code_text.window_text`'s lines, joined by newlines), the 10 s before it, their word counts and the text's `sha` (the first 16 hex digits of its sha256);
- `group`: the same of the group microphone alone;
- `timed`: the words that start from 2 s before the window to 2 s after it, as offsets from its start, with their ends and microphones;
- the table's speech measures, as the sensing audit freezes them;
- with `--audit-content-scores ARM=FILE,...`, each arm's scores of that text.

The file is readable by its owner only. `--audit-asr-events` names the version's events: `influx` (InfluxDB, through `--influx-config`) or a file pattern with `{sid}` that holds JSON lines (gzipped or not, as a backup writes them) or a JSON list. These checks run at every freeze:

| Check | Fails when | Then |
|---|---|---|
| The count | the events are not as many as the version's table was fused from, or the table changed since it was fused | the session is not frozen, unless `--audit-allow-drift`, which records it |
| The reference texts | with `--audit-asr-reference FILE` (JSON lines of session, window start, source, `cur` and `prev`), a window's text or the 10 s before it is not its row of source `all`, or has no row | the session is refused; no flag overrides it |
| The content scores | in an arm whose file has a `cur_sha` (or `cur_sha256`) column, a row has none, or one that is not the frozen text's `sha` | the session is refused; no flag overrides it |
| The timed words | the timed words that start inside the window are not the words of its stamped lines, so the reader has drifted from `code_text` | the session is refused; no flag overrides it |

At sampling, a session that fails a check refuses the whole sample and nothing is written, except that, without a source audit, a session that fails the count is left out. At `--audit-freeze`, a session that fails is refused alone, in a file that says why. A refused session can be frozen again, and its refusal stays in the log; `reported` is frozen again only through the same checks, since `--audit-freeze` refuses it without `--audit-asr-reference` and an arm with `cur_sha`. An arm without a `cur_sha` column, such as a cloud model's, can be frozen beside that arm; its rows are matched by their window alone.

### Transcription steps

| Step | Command | What it does |
|---|---|---|
| 1 | `--audit-sample ID --audit-task transcript --audit-version rerun --audit-asr-events influx` | Chooses the sessions, draws the windows ([below](#the-transcription-sample)), writes the plan, the views and the designs, freezes `rerun` and writes `campaign.yml`. The plan records the normaliser (`audit_text`, its version and sha256) and the declared analyses. |
| 2 | `--audit-freeze ID --audit-version reported --audit-asr-events PATTERN --audit-tables PATTERN --audit-asr-reference FILE --audit-content-scores ARM=FILE,...` | Freezes the other version for the same windows, checked against the texts the content model read and its scores' `cur_sha`. Both flags are required. |
| 3 | `--audit-render ID` | Cuts every window's clips, of every sweep, with ffmpeg alone (no `cv2`, so any environment with ffmpeg works), and copies the reveal into the views. It refuses until each recording's two versions are frozen, or refused, by logged steps. |
| 4 | `--campaign artifacts/runtime/audit/ID --log-note TEXT` | The notes below. |
| 5 | `--campaign artifacts/runtime/audit/ID --issue-token NAME --token-scope audit` | A one-time link for an auditor; `--token-subset reliability` for the second auditor. An open audit skips this step. |
| 6 | `--audit-estimate ID` | The hours, from 240 s per transcription and 45 s per reveal rating, or from the practice answers once there are some. The reveal's time stays assumed, since practice items are never revealed. |
| 7 | `--audit ID --bind <tailnet address> --allow-from <auditor machines>` | Serves the page, sweep 1 only unless `--audit-sweeps 2`. With `--audit-open`, no link is needed. `--audit SENSING --audit-with ID` serves the sensing audit and this one together, at one address ([below](#serving-both-audits-at-one-address)). |
| 8 | `--campaign artifacts/runtime/audit/ID --anchor` | Prints only hashes, as for the sensing audit. |
| 9 | `--audit-close-blind ID --audit-auditor NAME` | Closes the primary auditor's blind pass ([above](#blindness-and-the-reveal)). The primary then rates the reveals. |
| 10 | `--campaign artifacts/runtime/audit/ID --close-campaign` | Closes the audit to answers, once the reveal ratings are in. |
| 11 | `--audit-export-references ID --audit-auditor NAME --audit-out FILE` | Writes the primary's transcripts for the content model to read again ([below](#references-for-the-content-model)). |
| 12 | `--audit-score ID --audit-version rerun` | Scores `rerun` against the answers, and `reported` beside it ([below](#transcription-scores)). |
| 13 | `--audit-purge ID` | Deletes every clip. |

Step 1 refuses any version but `rerun`, since the declared analyses take it as the primary and the reveal shows it, and it refuses without `--audit-asr-events`. Without `--audit-tables`, a version's tables are each session's current fused table. A transcription audit is always blind: `--audit-mode` does not apply.

The sessions are those of the source audit given by `--audit-windows-from` (or those of them `--audit-sessions` names), and every one of them must load, since the source's windows are never thinned. Without a source, they are those of `--audit-sessions`, else every session less `--hold` unless `--audit-include-held`, and a session without video, a fused table, the version's events, or a group microphone or a mix is left out with the reason.

**Notes to log before the first answer** (`--log-note`):

- the roles: the primary auditor and the reliability auditor. The scorer finds this note by the word "role", for example `--log-note "roles: primary NAME, reliability NAME"`, and every score's header warns when no such note precedes the first scored answer;
- each auditor's first language, as they state it, whether they have seen the ASR text of these recordings before, on the coding page or elsewhere, and whether they answered the sensing audit the scores compare with, and under which name;
- the sha256 of the [Transcription guide](transcription_guide.md) the auditors were given, and again if it changes after the practice;
- the listening setup (headphones, a quiet room);
- the instructions to the auditors: no automatic transcription or translation tool, no other page of the server during the audit, and no talk about scored items with each other.

!!! warning "The ASR text on other ports"
    The tool cannot stop an auditor from opening the default coding page or the dashboard, which show the ASR text. The one-port rule of [Network isolation](#network-isolation), or the instructions above when the operator does without it, keeps the blind pass blind. Record which in the log.

**Practice.** Both auditors answer the practice items first. They then meet once to compare their practice transcripts with each other, never with the system's text, and settle questions about the conventions. The guide may be revised then; log its new sha256 before the first scored answer. The normaliser is never changed after the sample.

### The transcription sample

All draws are seeded (`--audit-seed`) and use window times only. A session's population is every window of the display grid inside its sound and every video, more than `--audit-margin` seconds (10) from either end, so a clip always lies inside the recordings. The lessons are those of `splits.lesson_key`.

| Part | Default | Flag | How it is drawn |
|---|---|---|---|
| Random windows (R) | the source's who-speaks windows; 10 per lesson the source drew none of | `--audit-windows-from SOURCE`, `--audit-speech` | With a source, a sensing audit, its who-speaks windows, aliases and recording order. Each window must start on the display grid (within 0.5 s), the source's microphone must still be the session's group microphone, and the window must lie inside the population; otherwise the sample is refused, never thinned. A lesson the source drew none of, or every lesson without a source, gets a simple random sample, shared over its sessions in proportion to their populations. |
| Sweep 1 | ranks 1 to 3 of each lesson | `--audit-first-ranks` | The other ranks are sweep 2. |
| Social windows (S) | 28 | `--audit-social`, `--audit-social-coders A,B`, `--audit-social-both` (0.65), `--audit-gap` (20 s) | Windows the named coders labelled `social`, by each coder's last label; a labels file a model wrote is refused. A share `--audit-social-both` of them is labelled social by every named coder, the rest by exactly one. Each lies at least the gap from every R and practice window, and from the S windows drawn before it in its session. A pool that runs out moves its shortfall to the other, printed and recorded. All are in sweep 1. `--audit-social 0` draws none. |
| Practice | the source's practice windows, and one window in the first session of each other kind of sound | `--audit-practice 0` turns it off | Without a source's, the first kind of sound (`group`, `dual`, then `mix`) gets 4 windows in its first session. Each lies at least the gap from every item. Practice comes first for every auditor and is never scored nor revealed. |
| Reliability subset | rank 1 of each lesson, and 25 % of the S windows | `--audit-reliability-ranks`, `--audit-reliability` | The second auditor answers these after the practice. |

**Ranks.** Within a lesson, the source's reliability windows (in a fresh lesson, two of its drawn windows) take the first ranks in a seeded order, and the rest follow in another. The source drew a simple random sample and flagged its reliability windows at random, so the ranks are a random order of a random sample: for every k, the windows of ranks 1 to k are a simple random sample of the lesson. An R window weighs its lesson's population over the lesson's R windows kept in the analysis. S windows carry no weight and enter only the analyses that are conditional on the coders' labels.

**Sweeps.** Every rank is drawn, frozen and rendered at once, but the server serves only sweep 1 unless it is started with `--audit-sweeps 2`. The items of a sweep not served are not in the sequence, not counted in the progress, and their clips answer 404. Sweep 2 can therefore be served later without drawing, freezing or rendering after any answer. A sweep beyond the plan's is refused, and so is `--audit-sweeps` for a sensing audit.

**Order.** The primary auditor gets the practice block, then sweep by sweep each recording's items of that sweep, R and S shuffled together, the recordings in the plan's order. The reliability subset gets the practice block, then its flagged items, the recordings in reverse order. The page starts at the first item not yet answered. Items are named `ID-<alias>-t-NNN`, practice items `ID-<alias>-pt-NN`. Each sweep's order is a seeded shuffle, so the R items an auditor answers in the page's order are a random sample of their lesson; an R item left unanswered before an answered one is a skip, which the scores count.

### The transcription page

The page has the player on the left and the form on the right. Under the video, a bar shows the context in grey, the shaded window and the tail, with the playhead; a click on the bar plays from that point to the end. The keys work outside the text fields. Enter and space on a button, link or the help's title that has the focus press, follow or fold it instead of playing. `Ctrl/⌘ Enter` saves from a focused button too, as it does from the text fields of the form.

| Button | Key | What it does |
|---|---|---|
| **Play window** | `space` | plays the shaded window, 10 to 20 s of the clip |
| **From 2 s before** | `a` | plays from 2 s before the window to the clip's end |
| **With the previous 10 s** | `p` | plays the context and the window, 0 to 20 s |
| **Back 2 s** | `b` | goes back 2 s and plays on to the end of the part playing |
| **Loop window** | `l` | plays the window again and again, until pressed again |
| **Speed** | `r` | 1, 0.75 or 0.5 times, the pitch kept |
| **Gain** | `g` | +0, +6, +12 or +18 dB, raised in the browser; the file is unchanged |
| **Microphone** | `c` | the table microphone or the personal microphones, at the same moment; `dual` sessions only |
| **Stop** | `s` | stops |
| | `t`, `n` | puts the caret in the transcript or in the note |
| **← previous**, **next →** | `←`, `→` | moves between items; unsaved changes ask first; nothing while an item loads or a save is sent |
| **Save** | `Ctrl/⌘ Enter` | saves, then opens the next item not yet answered, after the last back to the first one still open |

In the transcript and the note every key types, except `Ctrl/⌘ Enter` (save), `Esc` (play the window) and `Alt ←` (back 2 s). The tag buttons **M:**, **T:**, **O:** and **?:** put the tag at the start of the caret's line, and the marker buttons **[x]**, **[bg]** and **{ }** insert at the caret (the braces round a selection). A key held down does not repeat. Choosing status `none` answers the other questions for no one and unticks overlap. An item is shown only once it has loaded: if it does not load, the page stays on the item before it and says so. Past the last item the page says how many items are not answered yet and offers the first of them; it thanks the auditor only when none is left. The help panel, "How to transcribe, and the keys", shows the guide's rules in English and the keys. The page keeps no draft in the browser.

The server checks every save:

- **The transcript** follows the grammar of `audit_text.parse_reference`: at most 12 lines of at most 300 characters, 2,000 in all; each line its tag (`M:`, `T:`, `O:` or `?:`), a space and its words; the words letters, spaces, `. , ? ! ' -`, the markers `[x]` and `[bg]` and guesses in braces. Digits, other brackets and punctuation, symbols, tabs and control or formatting characters are refused, with the line and the rule named. The transcript is kept as the grammar cleans it: NFC, each line trimmed and its runs of spaces made one, typographic apostrophes made `'`, empty lines dropped.
- **The status** agrees with it: `none` with an empty transcript, `unintelligible` with only `[x]` and `[bg]`, `speech` with at least one word. Status `none` goes with who speaks `none`, and the other way round, and with no adult, no talk among the group's pupils, none of another group's and no overlap. Who speaks `adult` goes with an adult question that is not `no`.
- **The listening record** (the parts of the clip played, the seconds played, the slowest speed, the highest gain and the microphones) covers at least 95 % of the shaded window, unless the item is flagged. A transcription saved before carries its record on, so a typo is fixed without listening again.
- **The blind pass** is still open: after the close, a transcription gets 409.

The listening record rests on what the browser reports. It keeps an auditor from saving a window unheard by accident; it is a convenience, not a control, and a paper should describe it so.

### Links and open mode

The transcription audit is served with links or open, exactly as the sensing audit ([The open audit](#the-open-audit)): the same name and scope rules, the same refusals and the same log. The blind close and the reveal are keyed by the name: the link's name, or in an open audit the name as typed, in NFC.

In an open audit the server trusts the typed name for the reveal, as it trusts it for everything else. A person who types, with the full audit, a name whose blind pass is closed sees that name's reveals, and could then transcribe under another name; the log keeps only the address and browser of each request. Use links when the second auditor's independence matters, for example for an agreement a paper reports. The scores of an open audit say it was open.

### Serving both audits at one address

`--audit SENSING --audit-with ID [--audit-open] --bind <tailnet address>` serves a sensing audit and a transcription audit from one server on one port (8766 unless `-p`). The sensing audit keeps the addresses and the page it has when it is served alone: `/audit`, `/api/audit/...`, `/audit/img/...` and `/audit/clip/...`, and the page's bytes are the same but for the link back to the entry (below). The transcription audit is served under `/t`: its page is `/t/audit`, and its routes, its clips and its redirects begin with `/t`. The server takes `/t` off before the transcription audit's own routing, and puts it into the page and into every address it sends.

The address alone (`/`) is the entry: one page that leads to every task, shown to everyone (it assigns nothing by name). Its tasks, in this order:

- **Code interaction windows**: the default coding page. The link is the address alone of port 8765 on the host the entry was opened at, which a coding server answers with the page whether it runs this version (through its redirect to `/code`) or an earlier one, so the two servers may be restarted in either order. A name typed on the entry that the server would take goes along as `?name=`, so the coding page takes it as if it were typed there and remembers it; without a name the coding page asks for one. The card says that the coding page shows the system's transcripts, and that an auditor of the transcription opens it only once the operator has closed their blind pass.
- **Each audit part**, in the order of the flags: "Identity, gaze and who speaks" (the sensing audit) and "Danish transcription", each with a line on what it asks.

In an open audit the entry asks the name first, with the rules the audit pages apply to a typed name: trimmed, in NFC, not empty, without a control, formatting or line-break character, and at most 100 bytes in its file name. Each part has two buttons, **full audit** and **reliability subset**. Enter checks the name and moves to the first button. A button saves the name and the scope in the browser, under the keys that part's page reads, hands them to that page in this browser tab, and opens it at `#start`. The page starts at once under the name and scope handed over, as **Start** would, and takes `#start` off the address; a name or scope the server refuses shows the page's form with the reason. The same address without them, typed, bookmarked or opened in another tab, shows the form. What follows `#` never reaches the server. The entry fills in this tab's name: the one a button last opened a part under in this tab, or the one a part's page last came back with through its **← All tasks** link. Without one, as in a new tab, the entry fills in the name kept for a part, the first part's first: the one its page last started with, or the one last chosen for it on the entry. When this browser last opened the chosen part under another name, the entry says so and writes nothing; a second press of the same button opens the part under the name typed. Answers are kept by name, so a name typed otherwise starts a second set of answers. With links, each part is a link to its page, and the auditor's own link names them, as before.

Each part's page has **← All tasks** at the left of its header, a link back to the entry; a page served alone has none. In an open audit the page hands the name it audits under to the entry, in this browser tab only, so the auditor who just left finds their own name there, also on a shared machine where another auditor used the other part. A key pressed while the link has the focus reaches none of the page's keys, and Enter follows it. The sensing page leaves through it at once, as it moves to another item. The transcription page treats the link as a move to another item: while a save is sent or the next item loads it does nothing, and when the item shown has changes not saved it first asks the same question; the browser does not ask a second time. The coding page links back to the entry when its server runs with `--entry-port` ([Coding campaigns and the audits](#coding-campaigns-and-the-audits)).

`--audit-code-port PORT` points the coding task at another port, and `--audit-code-port 0` leaves it out. Give it the coding server's `-p` when that is not 8765. The flag goes with `--audit-with`; a port below 0 or above 65535, or the audits' own port, is refused. The start message prints each part with its address, then the entry's address and the coding page's port, and each part's start line records the port as `code_port` (0 when the link is left out).

!!! warning "The coding page and blind auditors"
    The coding page shows every coder's labels and, with its Transcript button, the system's text of every window. An auditor whose blind transcription pass is still open must not open it, and the entry's card says so. The entry only links to the coding page; whether an auditor's machine reaches port 8765 is set by the network ([Network isolation](#network-isolation)). Where the one-port rule applies, the link cannot connect: serve the audits with `--audit-code-port 0`. Where the operator relies on the instructions to the auditors instead (see the notes to log in [Transcription steps](#transcription-steps)), the link works: serve with `--audit-code-port 0` unless the auditors also code, and log the choice with `--log-note`. The start line records the port either way.

Each part stays an audit of its own:

- **Files.** Each part keeps its own folder, plan, `campaign.yml`, request log and answers files. A request to one part opens no file of the other (only a link at `/c/<token>` reads both audits' `campaign.yml`, to find its part; see Links), so serving the transcription audit beside the sensing audit leaves the sensing audit's frozen plan and its answers as they are. The transcription audit draws its own sample at `--audit-sample`.
- **Names and scopes.** The rules of names and scopes hold in each part apart. A name may answer both parts, with the full audit in one and the reliability subset in the other. The browser remembers the name and scope of each part apart, and the entry writes them for the part chosen.
- **Mode.** `--audit-open` serves both parts open; without it, both parts take links. Each part is refused as it is when served alone: a closed audit is not served, a part served with links needs a link issued in its own folder, a part with answers typed on the open page is served open only, and a name whose answers came through a link is refused on the open page.
- **Links.** A link of either audit works at the same address (`/c/<token>`). Such a request belongs to neither part until the server has read both audits' `campaign.yml` to find the one that holds the token; it writes nothing of either, and the audit that holds the token claims and logs the link. A link under `/t/c/<token>` goes to the transcription audit without that search. The two parts' cookies have different names, so one browser may hold a link of each. In an open audit no link is served: `/c/<token>` shows a page that sends the auditor to `/`, and `/t/c/<token>` one that sends them to `/t/audit`.
- **Sweeps and closes.** `--audit-sweeps` applies to the transcription audit, and it is refused when neither audit is one. `--audit-close-blind` closes a blind pass while the server runs, as it does for a transcription audit served alone.

The `--audit-with` audit is a transcription audit. A sensing audit's page asks for `/audit` and `/api/audit/` itself, so a sensing audit is always the `--audit` part, and two sensing audits are served by separate servers. Two transcription audits may be served together; their titles on the entry then carry their audit ids. The same audit is refused as both parts, also under two ids that differ only in case where the file system opens them as one folder.

Each part's request log gets its own start line and its own stop line. The start line is the one the audit's server writes when it serves it alone, with three fields more: `prefix`, the part's own (empty for the `--audit` part, `/t` for the other), `together_with`, the other audit's id and prefix, and `code_port`, the coding page's port the entry links to (0 for none). It names the modules of both parts, since one server runs them. Each request is logged in the log of the part that answered it, with the path as that part routes it (without `/t`), and the entry (`/`) in the log of the `--audit` part. `--verify-log`, the scorer's check for an open serve and the scores therefore read each audit as after a serve of its own. Ctrl-C stops the one server and writes the stop line in both logs.

### Transcription answers

The answers go to `artifacts/<session>/audit/<ID>/answers/<auditor>.jsonl`, one line per save, with the fields of the sensing audit's lines, the sweep and codebook version 2. The file is readable by its owner only, since it holds transcripts. Of an item's phase, the last line counts. A `transcribe` answer holds:

```json
{"status": "speech", "transcript": "M: ...\nT: ...", "overlap": false, "who": "member",
 "adult": "no", "peer": "task", "other_offtask": "no", "flag": false, "note": "",
 "listening": {"seen": [[8.0, 22.0]], "played_seconds": 14.0, "slowest_rate": 1.0,
               "highest_gain_db": 0.0, "channels": ["main"], "target_cover": 1.0}}
```

A `reveal` answer holds `gist`, `invented`, `flag` and `note`.

### References for the content model

`--audit-export-references ID --audit-auditor NAME --audit-out FILE` writes one JSON line per scored item NAME transcribed. It holds the last transcription saved before NAME's close (every line when NAME is not closed, which the command says), its status and flag, the request's sequence number, each frozen version's text, the 10 s before it and its `sha`, and the transcript's LLM form: the lines' words without tags, markers or fragments, a guess without its braces, the auditor's casing and punctuation kept, one line per turn. Notes are left out. The file is written with mode 0600, never over an existing file, and its sha256 is logged.

Run the frozen content scorer on these references, on the machine that holds them, to make the file of `--audit-content-scores-ref`.

??? info "Details: the file of re-scored references"
    `--audit-content-scores-ref ARM=FILE,...` reads per arm a CSV with the columns `item`, `version`, `variant`, `context`, `cur_sha`, `prev_sha`, `ref_sha`, `request_seq` and the content model's letter probabilities, and a `FILE.meta.json` beside it (`scorer`, `scorer_sha256`, `normaliser_sha256`, `model`, `revision`, `input_sha256`, `rows`). Every hash is the first 16 hex digits of a sha256.

    | Variant | Text | Context before it | Context |
    |---|---|---|---|
    | `ref_ctx1` | the transcript's LLM form | the version's 10 s before | 1 |
    | `ref_ctx0` | the transcript's LLM form | none | 0 |
    | `asr_ctx1` | the version's text | the version's 10 s before | 1 |
    | `ref_n1` | the transcript in N1's form | the version's 10 s before, in N1's form | 1 |
    | `asr_n1` | the version's text in N1's form | the version's 10 s before, in N1's form | 1 |

    A row counts only when its hashes are those of the counted transcript (and its request seq) and of the version's text. A variant with a row that fails, or with two rows of one item, is refused with the count, and the N1 variants are refused too when the meta names another normaliser than the plan's.

### Transcription scores

`--audit-score ID --audit-version rerun [--audit-auditor NAME] [--audit-boot 10000]` writes the scores to `artifacts/runtime/audit/ID/scores/rerun_<time>/`. `rerun` is the declared primary. The other version, when it is frozen and passes its checks, is scored in the same run beside it, with the paired differences. `--audit-version reported` puts `reported` first instead: the folder is named after it, and every header calls it the primary.

Before scoring, the scorer checks the request log, the frozen files and the tables as for the sensing audit, and that `audit_text.py` is the file the plan froze; `--audit-allow-drift` scores with another, named in every header. `reported` counts only where its freeze checked every window against the reference texts and an arm's `cur_sha`: otherwise it is refused when scored and left out, named, when scored beside, and every header says what it was checked against. The blind closes are the logged ones; where `blind_closed.json` differs from them, every header names it. The scorer refuses before the primary auditor's blind close, unless `--audit-interim`, which is logged and named in every header.

What counts:

- **The primary auditor** is `--audit-auditor NAME`, else the plan's (`--audit-primary NAME` at sampling, optional), else the one auditor whose link has no subset or, in an open audit, the one name that chose the full audit. The names that chose the reliability subset give the agreement. Other names are counted and left out. Answers of two auditors are never averaged or merged.
- **An item's transcription** is the last line saved before its auditor's blind close. Lines saved at or after it are left out and counted. **A reveal rating** counts only when saved after the close.
- **Left out and counted:** practice items; items not rendered, not answered or flagged; items the version holds no text of; transcripts the grammar now refuses or their status disagrees with; and the skips, an R item left unanswered before an item answered after it in the order the page served its recording.

Both texts are normalised by **N1** (`audit_text.normalise`) and aligned word by word, which gives the reference words N, the hits, and the substitutions S, deletions D and insertions I, per tag as well. The WER is Σ w·(S + D + I) / Σ w·N over the items with N ≥ 1, weighted by the R weights, with the unweighted value beside it. The CER is the same over characters.

??? info "Details: N1, N0 and the aligner"
    - **N1**, on both sides: fragments (`mi-`) out; NFKC and casefold, then ä→æ, ö→ø, é, è and ê→e, á and à→a, ü→u, with æ, ø and å kept; a thousands dot out and runs of digits up to 9999 spelled as Danish cardinals; punctuation and symbols a space, apostrophes deleted; the variant map (`hva` is `hvad`, `ok` is `okay`); the fillers deleted. On the reference, the tags, `[x]` and `[bg]` go first, and a guess counts as its words.
    - **N0**, a sensitivity: the fragments, NFKC with casefold and the punctuation step only.
    - **The aligner** is a Levenshtein alignment with unit costs that breaks its ties the same way every time: a hit or a substitution, then a deletion, then an insertion.
    - The plan freezes the normaliser's version and sha256 at sampling, and the docstring of `audit_text.py` lists every step.

**The content scores.** A content arm is a CSV of a content model's scores per window: session, window start, source, context, `no_text`, the letter probabilities and `cur_sha`. An arm frozen with a version (`--audit-content-scores` at sampling or at the freeze) is read from the frozen file. For the version scored, `--audit-content-scores ARM=FILE,...` at scoring adds arms. Each of their rows' `cur_sha` (or `cur_sha256`) must be the frozen text's `sha`: a file without that column, or with a row that differs, is refused with the count, and no flag overrides it. The arm named `qwen` gives the primary content analyses (P2 and P3), and the others are reported apart. A window the content model was not asked about (`no_text`) scores 0.

| Check | Measures |
|---|---|
| Accuracy (P1, S6) | the WER of `consumed` on R items, with its substitution, deletion and insertion shares, the CER, the reference words and the hypothesis tokens; the content words both texts share (N1 less the Danish stopwords and the response words), as precision, recall and F1 |
| Detection (S1) | the miss rate (no text where words were written), the phantom rate (text where no one spoke, and apart where speech was unintelligible), hypothesis tokens per minute of such audio, repeated 3-grams |
| Sources (S4) | recall per tag (`M`, `T`, `O`, `?`), the share of the hits per tag, the WER by who speaks, teacher recall against member recall |
| Group source (S2) | the WER of the `group` text; the WER by kind of sound; in `dual` sessions `consumed` against `group`, and the insertion shares |
| Words feature (S5) | the table's `words` against the reference words (N0); `speech_ratio` against the status |
| Content (P2, P3, S7, S12) | the within-lesson concordance C of each score with the blind answers (`peer_other`: peer other against task; `teacher`: adult yes against no; `peer_task`: peer task against none), with the pooled AUROC beside it; from the re-scored references, C_ref − C_ASR and the transfer ratio; the scores' agreement between the two texts (Lin's CCC, Spearman, MAE, κ) |
| Who speaks (P4, S10) | the primary's shares of who speaks on R items with measured speech, in the sessions with a group microphone and in all; with `--audit-compare-with ID`, a sensing audit joined on the session and the window start (its log verified, read only), the paired difference from its primary, κ, confusion matrices and, for a name in both audits, test-retest; the transcript's main tag against who speaks, on the items whose transcript has a word |
| Reveal (S11) | `gist` and `invented`, as ratings of the version the reveal showed (`rerun`), whichever version is scored |
| Coders' labels (S8) | the heard topic (`peer`) against each social coder's labels, as shares, κ and C |
| Social stratum (S13) | accuracy and detection on the S items alone |
| Versions (S3) | the other version's accuracy, detection and group source, and the paired differences, each on the items its measure keeps in both versions |
| Sensitivities | N0; a 0.5 s edge tolerance (timed words near an edge optional); split and merge; the best order of overlapping lines; without the items with `[x]` or overlap; with the insertions of empty windows; the lesson macro mean; sweep 1 alone; without the sessions `--audit-exclude-sessions` names (ids or aliases) |
| Agreement (S9) | the reliability auditor against the primary on the items both answered: the WER and CER each way, the system's WER against each and its excess over that floor, κ for status, who, adult, peer, other_offtask and overlap |

Every pooled value has a lesson-cluster bootstrap interval (10,000 draws unless `--audit-boot`), every difference paired draws, and each primary a leave-one-lesson-out range. Per-lesson rows give counts and point values only, and no p-value is given. The decision bands of the plan's declared analyses stand beside the rows they apply to.

The outputs hold counts only, never a word of a transcript or of the system's text: `report.txt`, `summary.json`, `units_<version>.csv` per item and source, `pooled_<check>.csv` and `lesson_<check>.csv`, `interauditor.csv`, `compare_<audit>.csv` and `content_<arm>.csv`. Every file is headed by the plan's and the anchors' sha256, the version, the primary and reliability auditors, the roles note or a warning, the blind closes, the lines left out after a close, the log's check and the deviations, and it says when the audit was open or the score is interim.

### Transcription audit files

```
artifacts/runtime/audit/<ID>/
  plan.json           task transcript, sessions, aliases, lessons and their populations, sizes, the social draw, the normaliser, declared analyses
  campaign.yml        the auditors' links (none in an open audit) and whether the audit is closed
  requests.jsonl      the request log: every step, each blind close and export, and the sha256 of what it wrote
  blind_closed.json   per auditor whose blind pass is closed, the close's sequence number and time (the page reads it; it must agree with the logged closes)
  media/<alias>/      the clips, <item>.mp4 and <item>_worn.mp4 (--audit-purge deletes them)
  render_index.json   per recording the clips cut and the errors, and the ffmpeg version
  scores/             the scorer's outputs
artifacts/<session>/audit/<ID>/
  view.json           what the server serves, with the reveal text once rendered (owner only)
  pipeline.json       the design: windows, strata, ranks, weights, sound and sources (never served)
  pipeline_<v>.json   a version's text and measures per window (owner only, never served)
  answers/<auditor>.jsonl
```

The rendered views, the frozen files, the answers and the exported references hold transcript text. Keep them, with the clips, on the machine that holds the recordings: a copy of a session folder carries its `audit/<ID>/` folder.

## A note on short flags

The campaign and audit flags share prefixes with older flags. Abbreviations that used to select an older flag are now ambiguous, and argparse refuses them: `--pre`, `--prep` and `--pr` (`--prepare-text`), `--t` (`--threads`) and `--i` (`--influx-config`). The transcription audit's flags do the same to `--audit-c` (`--audit-check-frames`), `--audit-f` (`--audit-freeze`), `--audit-i` and `--audit-in` (`--audit-include-held`), `--audit-pr` (`--audit-practice`), and to every abbreviation of `--audit-reliability` from `--audit-rel` on. Spell flags out in scripts. No script in the repository uses these abbreviations.
