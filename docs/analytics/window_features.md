# Window features

`mmla ses-fuse` joins what the ASR, IPS and VFA pipelines stored for a session into one table, the fusion table: one row per time window (10 s by default), with the speech, space, body, gaze and action features side by side and a coverage count per modality. Use it to analyse a session across modalities; the dashboard's video job builds the same table, with the default settings, for its [Analysis page](../dashboard/index.md#analysis).

## What the table holds

| Block | What it gives | Events |
|---|---|---|
| [Window](#window-columns) | the window's index, start and end, the keys other tables join on | the session's span |
| [Speech](#speech-columns) | speech and silence, named speakers, talk spurts, words, anonymous speaker turns | `asr_recognition`, `asr_transcription` |
| [Space](#space-columns) | presence, the path walked, distances, who faced whom | `ips_translation`, `ips_rotation`, `ips_relation` |
| [Body and gaze](#body-and-gaze-columns) | head turn, where gazes landed, joint and mutual attention, hand distance, hand activity in body units | `vfa_features` |
| [Actions](#action-columns) | the VLM's action label per person | `vfa_action` |
| [Seats and non-members](#seat-and-non-member-columns) | untagged bodies at the persons' seats; someone outside the group at the group's table | `vfa_features` |
| [Content](#content-columns) | a transcript reader's scores: teacher talk, peers on the task or on something else, no text; appended to a copy of the table, not written by `ses-fuse` | `asr_transcription`, through the reader |

Persons are tag ids (`p<tag>_...`) and pairs are sorted tag pairs (`pair<a>_<b>_...`). Bodies the pose model saw without a tag count only in the seat trace. A cell is empty when its modality gave nothing to judge in the window.

## What you need

- **The session's events**, in InfluxDB ([Databases](../database.md#influxdb)) or in the measurements folder that **Sessions → Export** writes, `artifacts/<session>/measurements` ([Export](../tui/sessions.md#export)).
- **A config with an `InfluxDB` section**, such as `pipelines/vfa-base/config.yml`, to read from InfluxDB.
- **The `uber-base` environment**: create it on the console's **Environment** tab ([Environments](../tui/environment.md#environments)), or by hand with `pip install -e '.[uber-base]'`.
- **The session's pupil tags**, the badges the group's own members wore ([Pupils](#pupils)).

## Build a table

1. **Declare the pupils.** Run `mmla ses-tidy <session-id> --pupils 0,1` once per session with the pupils' tags; it writes them into the session's manifest ([Correct a session](../tui/session-tools.md#correct-a-session)). Undeclared, the pupils are the tags up to 12.
2. **Run the fusion**, from InfluxDB or from an export:

    ```bash
    conda activate uber-base
    # from InfluxDB, through any config with an InfluxDB section
    mmla ses-fuse -c pipelines/vfa-base/config.yml -sid <session-id>
    # from an export
    mmla ses-fuse --measurements artifacts/<session-id>/measurements
    ```

3. **Check the output.** The command prints the pupils and where they came from, then the windows, columns and events it read, and the table's path: `artifacts/<session>/analysis/features/<session>_window_features.csv`. The [analysis record](#analysis-record) beside it says what the table was made from and with.

??? info "Details: the export files read"
    - The command reads `<session>_<suffix>.json` from the folder: `speaker_recognition`, `speaker_transcription`, `action_recognition`, `features` (the `vfa_features` events), `badge_translation`, `badge_rotation` and `badge_relation`. The plural `badge_*s` names are read too. A type without a file is empty.
    - With `-sid`, only that session's files are read. Without it, every export file in the folder is read, and the session is told from the file names, else from the folder's parent when the folder is named `measurements`.

### Command options

| Flag | Default | What it does |
|---|---|---|
| `-p`, `--project_dir` | the working directory | the project directory, whose `artifacts/` holds the session |
| `-c`, `--config_path` | | a config with an `InfluxDB` section, to read the events from InfluxDB; needs `-sid` |
| `-sid`, `--session_id` | | the session |
| `-md`, `--measurements` | | an export's measurements folder, read instead of InfluxDB |
| `-w`, `--window` | `10` | the window length, in seconds; above 0 |
| `-st`, `--step` | `10` | the step between windows, in seconds, above 0; equal to `-w` for windows that do not overlap |
| `-tags`, `--participants` | the tags the events hold | comma-separated tag ids to build columns for; they are then the session's pupils too |
| `-hr`, `--hand_relabel` | `true` | remake the gaze targets and hand distances with the current hand circle ([The hand circle, remade](#the-hand-circle-remade)) |
| `-tm`, `--tag_memory` | `60` | the tag memory, in seconds ([Tags along the tracks](#tags-along-the-tracks)); `0` keeps every kept tag and carries reads without a limit |
| `-fr`, `--face_refusal` | `60` | face verdicts in a row that take a remembered tag off its track ([Face refusals](#face-refusals)); `0` leaves the face checks unread |
| `-pr`, `--path_rule` | `true` | count the path on the floor, smoothed ([The path](#the-path)); `false` sums every raw step between the window's own positions, in 3D |
| `-js`, `--joint_split` | `false` | split joint attention by where it met, and add the non-member columns ([Where joint attention met](#where-joint-attention-met)) |
| `-cams`, `--cameras` | every camera | comma-separated camera ids: keep only these cameras' frames of every VFA frame set, and drop a set left with none |
| `-fs`, `--frame_sets` | every frame set | `PERIOD:KEEP`, with 1 ≤ KEEP < PERIOD: number the VFA frame sets in time order and keep set k when k mod PERIOD < KEEP; `4:2` keeps two sets in every four, `2:1` every other one |
| `-o`, `--out` | `artifacts/<session>/analysis/features/<session>_window_features.csv` | where to write the table: `.csv`, else JSON lines |

The switches take `true` or `false`.

??? info "Details: filtered tables with `-cams` and `-fs`"
    - Both filter the VFA frame sets alone, before anything else, `-cams` first. Every setting counted in frames or seconds stays the same.
    - The windows stay those of the table fused without the filters, so the two tables line up row by row.
    - A filtered table never goes into `artifacts/<session>/analysis/features/`: give `-o` outside it. The command refuses otherwise.
    - A camera id is the `camera` a frame carries, else its angle key ([Body and gaze columns](#body-and-gaze-columns)). `-cams` refuses a camera with no frame in the session and lists the cameras the frames carry.
    - The analysis record says what each filter kept (`cameras`, `frame_sets`, `window_span`). A filtered run whose record cannot be written exits with an error.

### Python options

`window_features(events, ...)` in `openmmla/analytics/fusion/window_features.py` builds the rows, and `write_table(rows, path)` writes them. Its keyword arguments include switches the command does not have:

| Argument | Default | What it does |
|---|---|---|
| `window`, `step` | `10.0`, `10.0` | as `-w` and `-st` |
| `participants` | the tags the events hold | the persons to build columns for, as `-tags` |
| `pupils` | the participants' tags up to 12 | the session's [pupils](#pupils) |
| `speakers` | the names the recognition events hold | the `spk_<name>_ratio` columns |
| `hand_relabel` | `True` | as `-hr` |
| `work_area` | `True` | label the [work area](#the-work-area); `False` leaves out `p<tag>_work_area_ready_ratio`, and the `work_area` share is 0 |
| `track_tags` | `True` | carry the tags [along the tracks](#tags-along-the-tracks); `False` also leaves out `n_vfa_propagated` |
| `seat_partners` | `True` | name an untagged gaze target at a missing pupil's seat as that pupil; `False` leaves every untagged target as someone else, leaves out `n_vfa_seat_partners`, and drops the seat tests of [who is outside the group](#who-is-outside-the-group) |
| `tag_memory` | `60.0` | as `-tm`; `None` for no limit |
| `face_refusal` | `60` | as `-fr`; `None` or `0` leaves the checks unread |
| `path_rule` | `True` | as `-pr` |
| `joint_split` | `False` | as `-js` |
| `span` | the session's span | the first and last moment the windows cover |

## How the frames are prepared

Before any window is cut, the fusion goes over the session's `vfa_features` frame sets once, in this order:

1. **Tag memory.** A tag the server kept on a track too long after its last read is taken off ([Tags along the tracks](#tags-along-the-tracks)).
2. **Face refusals.** A remembered tag the face check kept refusing is taken off its stretch of the track ([Face refusals](#face-refusals)).
3. **Hand circles.** The gaze targets and hand distances are made again with the current hand circle ([The hand circle, remade](#the-hand-circle-remade)).
4. **Work area.** An `elsewhere` gaze inside the camera's work area becomes `work_area` ([The work area](#the-work-area)).
5. **Propagation.** The tags are carried along the tracks to the frames around their reads ([Tags along the tracks](#tags-along-the-tracks)).

The [seats](#seats), the pupils' tracks and the pupils' sizes are learned from the tags the server gave, after steps 1 and 2, never from propagated ones.

The windows run from the session's earliest event until its last moment is covered. A frame set or an action label belongs to the one window its moment falls in; a span (an ASR bucket, a transcript chunk, an IPS second) counts in every window it overlaps.

## Tags along the tracks

The features endpoint keeps a read tag on its track from then on (`tag_match: track`, see [Identity](../pipelines/vfa/pose-and-gaze.md#identity)). A track that lost its person and was picked up by another body carries the tag on to the wrong person. The fusion therefore limits the tags and then extends them:

- **Tag memory.** A kept tag is trusted for 60 s (`-tm`, `TAG_MEMORY_SECONDS`) after the track last read it. On later frames it is taken off, and the person is an untagged body again (`person_id` `track_<id>`).
- **Propagation.** A tracked person without a tag, on a track that read a tag at most 60 s before or after, takes the tag of the track's nearest read and is marked `tag_match: propagated`. So the frames before a track's first read are named too.

A propagated person counts like a tagged one in every column, and `n_vfa_propagated` counts them per window; the seat trace counts only the bodies still untagged after propagation. The work area, the seats and the pupils' tracks never learn from propagated tags, so a track carried to the wrong person cannot move them.

??? info "Details: tag memory and propagation"
    - The memory also takes off a kept tag the track never read, or whose last read was another tag.
    - When a tag is taken off or given, the frame's pairs and gaze targets follow the new name.
    - Propagation runs per camera over the whole session. A track that read two tags is split between them: each untagged frame takes the nearer read, the earlier one on a tie.
    - A read tag, and a tag the server kept within the memory, are never changed. A person the face refused a tag is never given it.
    - A tag is not given in a frame where another person already carries it, nor to two tracks in one frame (neither gets it). A track id that appears twice in one frame names nobody.
    - A track id not seen for more than 60 s (`TRACK_GAP_SECONDS`) counts as a new track: ByteTrack keeps a lost track for 30 frames, so a longer absence means the id was handed out again.
    - A track that never read a tag stays untagged, and so does a stretch of a track more than 60 s from its nearest read.
    - A track the server split ([Tracking](../pipelines/vfa/pose-and-gaze.md#tracking)) has a new id, and reads are carried along one id only, so they cross neither the split nor a frame whose tag the server withheld.
    - Events without `track_id` get no propagation and no `n_vfa_propagated` column.

### Face refusals

When the server's appearance checks are on, it checks each person whose track remembers a tag against that tag's gallery and records the verdict in `reid.tag`: the `kind` of check (`face` or `colour`), its `score`, the `verdict` (`same`, `different` or `unknown`) and the `tag_id` it checked ([Appearance checks](../pipelines/vfa/pose-and-gaze.md#appearance-checks)). The fusion reads the face verdicts alone.

On one camera's track, a run of at least 60 face verdicts (`-fr`, `FACE_REFUSAL_FRAMES`) that call the person someone else than the remembered tag takes the tag off the track from the run's first such verdict to its last, frames without a face included. The persons of that stretch are marked `tag_refused`, and propagation never gives them that tag back. A single `different` verdict, or a short run of them, is often wrong, so only a long run acts.

??? info "Details: what ends a run"
    - A read of the tag (torso or box), or a `same` face verdict on it, ends a run. A frame without a check, an `unknown` verdict or a colour verdict does not.
    - Runs are counted per track as propagation counts tracks: a track id not seen for more than 60 s starts a new track, and a frame where the id appears twice is skipped.
    - The fusion is offline, so it sees the whole run before it decides.
    - Those of the stretch who still carry the tag become untagged bodies (`person_id` `track_<id>`), with the frame's pairs and gaze targets renamed. Propagation may give them another tag, and another person of the frame may then take the refused one.
    - With the server's default `tag_check_acts: false`, nothing on the server acts on the verdicts. The fusion ignores every other `reid` field.
    - The dashboard's video report applies the tag memory and the face refusals too, before it counts the looks between pupils.
    - A session whose persons carry no checks gives the same table with `-fr 0` and without it.

## Pupils

A partner is another pupil of the session. The pupils are the first of these that exists:

1. the `-tags` given to `mmla ses-fuse`;
2. the `pupils` the session's `manifest.json` declares (`mmla ses-tidy --pupils`);
3. the tags up to 12, the IPS trust bound: a higher tag is a misread badge.

The command prints them with their source, and the analysis record keeps them (`pupils`, and `pupils_source`: `tags`, `manifest` or `trust bound`). The dashboard's video job reads neither `-tags` nor the manifest for its table, so its `window_features.csv` always takes the tags up to 12.

A gaze target is a partner when its `person_id` is a pupil's tag after the tags were carried along the tracks. A target that stays untagged is still a partner when it stands at the [seat](#seats) of a pupil whose tag that camera's frame does not hold: the badge was not read, and nobody else sits there. `n_vfa_seat_partners` counts the gaze frames named this way. Every other face or hand (an untagged body, the teacher, another group, a misread badge) is `other_face` or `other_hands`.

!!! note
    A pupil with an unread badge away from their seat still counts as `other_*`. The seat rule lessens this error but does not remove it.

Who counts as outside the group for `_other_hands_near_ratio`, the joint split and the `nm_*` columns is ruled on separately: see [Who is outside the group](#who-is-outside-the-group).

## Seats

A person's seat on a camera is the median centre of their own tagged boxes there, from at least 10 boxes. Only the boxes the server tagged count (read or kept on the track), not propagated ones, so a track carried to the wrong person does not move a seat. A body is at the seat when its box centre lies within half the person's median box width of it.

Seats are learned once per session, from the whole session, and ignore camera moves. The [pupils](#pupils), [who is outside the group](#who-is-outside-the-group) and the [seat trace](#seat-and-non-member-columns) use them.

## The work area

A gaze on the shared table, a microscope or a tablet lands on no face and no hand, so the server calls it `elsewhere`, together with a gaze at the floor or at the next table. The fusion learns, per camera and frame size, where the pupils' hands have been, and calls an `elsewhere` gaze that lands there `work_area`. Faces, hands, zones, `out_of_frame` and `unknown` keep priority: only `elsewhere` is relabelled.

| Rule | Value |
|---|---|
| Area | the box between the 2.5th and 97.5th percentile (nearest rank) of the pupils' hand centres on each axis |
| Padding | the median hand radius, plus the gaze tolerance every target gets (the frame's longer side / 64); clipped to the frame |
| Ready | from the 200th hand sighting: about 40 to 60 s with two or three pupils at one frame set a second |
| Learns from | the pupils' hand circles, from wrists seen with a confidence of 0.3 or more, by the tags the server gave (read or kept); never propagated tags, never the teacher's hands |
| Time | causal and cumulative: a frame is labelled with the hands of the frames up to and including it, never later ones |

`p<tag>_work_area_ready_ratio` says how much of a person's gaze fell on cameras whose area was ready. The fusion writes the area's box into each frame it labels, and keeps a frame that already carries `work_area` as it is.

??? info "Details: limits of the work area"
    - In 2D the area is not free of faces: a partner's face above the table falls inside it. The relabel works because faces and hands are scored first.
    - A camera at the edge of the group that sees few pupil hands is rarely ready, so its persons' `work_area_ready_ratio` is low.
    - On a camera close to the table, the area covers a large part of the frame.
    - The padding by the hand radius matters on a microscope side camera, where the eyepieces sit just beyond the hands on the focus knobs.
    - Early areas are smaller, which errs toward fewer relabels.
    - Each camera and frame size has one area for the whole session, so a camera moved during a session stretches it.
    - Code: `openmmla/services/vfa/work_area.py` and `label_work_areas`.

## The hand circle, remade

The VFA server scores gazes against a circle around each hand ([Hands](../pipelines/vfa/pose-and-gaze.md#hands)), and a frame made with version 1 of the circle (any frame without `scoring`) gives other targets and hand distances than the current version 2. So the fusion makes every frame's gaze targets and pair hand distances again with the current circle, from what the frame stores. No video is replayed.

The remake reads the keypoints and boxes, `face_bbox`, the gaze point and `inout`, the zones, the frame's size, and the thresholds the frame states in `scoring`, else the server's defaults: `keypoint_confidence` 0.3 and `inout_threshold` 0.5. The rest of the frame is kept, the gaze distances too. With version 2, more gazes land on hands, a person's own or a partner's, and fewer on the work area.

`-hr false` keeps the stored targets and distances; the work area and the hand columns then read the circle most stored frames were made with (version 1 when none says).

??? info "Details: frames left as stored"
    - A frame without a size, or with a person without keypoints, a box or a name; a frame that already carries `work_area`; and a frame the server already made with the circle asked for (its `scoring` says so).
    - The target of a person with a gaze point but no face box.
    - Remade with the circle it was made with, a frame gives back its stored answers, except where the rounding of the stored numbers (coordinates ±0.05 px, confidences and `inout` ±0.0005) puts a value onto a threshold, such as a keypoint confidence of 0.3 or an `inout` of 0.5.
    - Code: `relabel_hand_circles`, with `features.relabel_answer` per frame.

## The path

`p<tag>_path_m` is how far the tag's badge moved on the floor in the window, in metres. A badge that stays put still moves a few millimetres to a few centimetres from one second to the next, and a sum of every raw step would count that jitter as walking. So the fusion counts each tag's path once over the whole session, before any window is cut:

1. **Floor.** Each position is laid on the floor plane of the dashboard's floor plan, normal to the badges' mean gravity in the session's `ips_rotation` records. Height does not count.
2. **Runs.** The positions are split into runs wherever two reads of the badge are more than 10 s apart (`PATH_RUN_GAP`). Within a run, each floor coordinate becomes the median of it and its two neighbours, so a single misread second disappears.
3. **Steps.** A step between two consecutive smoothed positions counts when it is at least 0.05 m long (`PATH_STEP_MIN`) and at most 1.0 m for each second between them (`PATH_STEP_MAX`). A faster step is a jump between cameras or a misread, and is left out, not cut down.
4. **Windows.** A counted step belongs to the window of its later position.

The cell is 0 when the badge was read and did not move, and empty when no step of a run ends in the window. Movement slower than 5 cm a second, such as shifting on a chair, is not counted, and neither is a walk faster than 1 m/s. `-pr false` turns the rule off ([Command options](#command-options)).

??? info "Details: edges of the path rule"
    - Leaving out height also leaves out a badge bobbing as its wearer leans, and the depth error along a pitched camera's axis that leaks into height.
    - With fewer than 30 rotations, or rotations that define no plane (a camera looking straight down), the plane is the main camera's own x-z plane, which is the floor plan's too for a camera hung upright or upside down. For a main camera on its side, the floor plan turns that plane by the camera's turn, which the fusion does not know, so there the two can differ.
    - Reads up to 10 s apart are bridged: a badge unread for 5 s that comes back 2 m away walked 2 m. No step crosses a longer gap.
    - A run's first and last smoothed position is the mean of two reads, as the floor plan smooths its tracks. A run's first and last step count only when the raw step between their two reads is within 1.0 m a second too, so a misread at a run's end that jumps further is left out. A slower misread there still counts by half (one 0.8 m off at the run's last read adds 0.4 m), and in a run of three reads, whose two steps are both end steps, in full.
    - A run of two reads counts nothing, and a read alone on one side of a gap at a run's end counts half of the move across it.
    - With overlapping windows, a step counts in every window that holds its later position.
    - The bounds and the smoothing are the floor plan's (`STEP_MIN`, `STEP_MAX`). The floor plan applies the bounds per 1 s window, breaks its runs at 1.5 s and does not check the raw step at a run's ends, so its `path_m` can differ a little where reads drop out: shorter where the fusion bridges a gap, longer by half of a fast misread at a run's end.
    - Code: `path_steps`.

## The hands in body units

The wrist speed and hand distances in frame widths do not depend on the camera's resolution, but they grow when a pupil sits nearer the camera, or the camera stands nearer or has a narrower field of view. The hand columns measure the hands in each pupil's own shoulder widths instead.

| Rule | Value |
|---|---|
| Step | two frame sets in a row, 0.75 to 1.5 s apart, that both hold the pupil, and only one body with the pupil's tag, on the same camera |
| Keypoints | wrists and shoulders seen with a confidence of 0.5 or more |
| Move | each wrist's displacement in the image, in the pupil's shoulder widths (the mean of the two frames'); a step takes the largest of its wrists' moves |
| Active | a wrist moved at least 0.16 shoulder widths |
| Still | every wrist seen moved less than 0.08 shoulder widths |
| Hand length | 0.46 shoulder widths, wrist to farthest fingertip; in hand lengths, still is under 0.17 and active from 0.35 |

A step between the two bounds is neither. Still is about how far the wrist of an arm that stays still in the image moves; active is twice that. Each step is taken on one camera, then the cameras are pooled, each step counting once.

The moves are in the image, not against the shoulders: a pupil who leans in with the hands resting on the table does not move them, while hands carried along with the body (shifting in the chair with the hands in the lap) do. An active step is the hands moving, whether the arm or the body moved them, not work on the artifact.

!!! warning "Session cadence"
    The bounds are moves in a 1 s step. A session at another cadence, such as a `keyframe_interval` of 0.5, has no hand steps, and its step columns are empty.

??? info "Details: limits of the hand columns"
    - These are 2D positions of the pose model's wrists. Hands close together are not hands touching, a hand at rest on the artifact can be working (fine manipulation with a still wrist reads still), and nothing here says what a hand holds.
    - At one frame set a second, a handover or a pointing stroke is seen once or not at all.
    - Two cameras that saw the same pupil on the same step can give it different statuses: a movement along one camera's line of sight is short in that camera's image, and each camera hides different wrists. A 10 s window pools at most nine steps per camera, a noisy reading of a pupil's hands.
    - A frame set where two bodies wear one tag (a track's remembered tag and a read one) gives no step, since the step cannot tell which body it follows.
    - The hand length is one fixed estimate from a whole-body pose model, not measured per pupil.

## Who is outside the group

Three kinds of columns ask whether someone outside the group is near: `p<tag>_other_hands_near_ratio`, the [joint split](#where-joint-attention-met) and the `nm_*` columns. A body that wears no pupil's tag counts as the group's own when it passes one of these tests; otherwise it is someone else, or for the joint split and `nm_*`, a non-member:

| Test | For `_other_hands_near_ratio` | For the joint split and `nm_*` |
|---|---|---|
| Its box overlaps a pupil's by an IoU of 0.5 or more, or lies 60 % or more inside it (a second skeleton, an arm seen apart) | the group's own | the group's own |
| It has a hand centred within 0.1 shoulder widths of a pupil's (a shared wrist) | the group's own | the group's own |
| Its box centre stands at a pupil's [seat](#seats) on that camera | the group's own, the pupil seen there or not | the group's own only when that pupil is missing from the frame, whatever tag the body wears |
| It is on a track the server gave a pupil's tag | at any time of the session | within 60 s |
| Its size is under 0.6 of the pupils' | not tested | a far body, at the next table: neither the group nor a non-member |

The size is the body's shoulder width against the median shoulder width of the pupils in the frame (else of the pupils on that camera over the session), or with its shoulders unseen, its box width against the pupils' median box width. A body at the group's depth reads about 1, and one about 1.7 times as far from the camera reads 0.6 or less. A teacher standing behind the table can read a little smaller than the pupils and still passes. A body whose size cannot be measured is neither the group nor a non-member.

A frame cannot tell who is a non-member when neither its pupils nor the pupils on that camera over the session showed a shoulder width. It then has no non-members, and the joint split counts a gaze on someone else there as on an outsider.

The rules look at boxes, seats and tracks, not at the picture, so what is left can still be a stray skeleton rather than the teacher or another group. A teacher leaning in at a present pupil's seat is a non-member for the joint split, which is who it is meant to find; a teacher standing where an absent pupil sat counts as the group in both.

## Joint attention against its own past

Two gazes that land within 5 % of the frame's width of each other are joint attention (`pair<a>_<b>_joint_attention_ratio`). Pupils who sit close look at the same table a lot, so this share partly measures seating. `pair<a>_<b>_joint_attention_baseline` is the pair's own recent rate of meeting:

- Each gaze point of a in the window (on a camera, in frame, at time t) is compared with b's gaze point on the same camera at the frame set nearest t − d, within 0.5 s, for d = 20, 30 and 40 s; and the same with a and b swapped.
- A hit is two points within 5 % of the frame's width. The baseline is hits over comparisons, pooled over the cameras, the lags and both directions, and empty below 10 comparisons.
- `_joint_attention_excess` is the window's ratio minus the baseline, from −1 to 1.

The lags sit past the decay of a joint episode and still within the same seating and task phase. An episode that began 20 to 40 s earlier is part of the baseline: the excess measures change against the pair's recent level.

??? info "Details: what is causal"
    - Only b's points from before the window are compared, whatever the window's length, so the baseline is causal in time.
    - Who a gaze point belongs to is not: it comes from the tags carried along the tracks over the whole session (a track takes the tag of its nearest read, even a later one) and from seats learned over the whole session. A badge read at minute 20 can change the baseline, and the partner and other split, of a window at minute 18. A server that keeps tags on tracks only forward would not reproduce these values exactly.
    - The work area is causal: it learns only from the tags the server gave, in time order.

## Where joint attention met

With `-js true`, the fusion splits each pupil pair's joint attention by what the two gazes met: the group's own things, or someone [outside the group](#who-is-outside-the-group). It is off by default. Each pupil's gaze first gets a referent:

| Gaze | Referent |
|---|---|
| a point within reach of a non-member: within the gaze tolerance of one of their hand circles, or of their face box grown by 20 % (without a face box, the circle of half their head width around the nose) | outsider, whatever its label |
| `other_face`, `other_hands` on a non-member, or in a frame that cannot tell | outsider |
| `other_face`, `other_hands` on a body that is the group's own | member |
| `partner_face`, `partner_hands`, `own_hands`, `work_area` | member |
| `other_face`, `other_hands` on a far body, on one whose size cannot be measured, or on no body of the frame; `elsewhere`; `zone` | none |
| `out_of_frame`, `unknown`, or no gaze point | none |

The reach rule overrides the gaze's label because the server gives a point to the nearest hand, so a teacher's hand beside a pupil's can leave the gaze named `partner_hands` or `own_hands`. A joint frame is an outsider frame when either gaze is on an outsider, since two points that close are on the same thing, and a member frame when both are on the group's own. The rest (met elsewhere, in a zone, or on a teacher the pose model did not see) stays in `_joint_attention_ratio` alone.

`_joint_member_baseline` uses the baseline's comparisons, a hit counting only when both gazes were on the group's own things. There is no outsider baseline: a demonstration lasts minutes, the gaze 20 to 40 s earlier is on the same teacher, and a baseline would absorb the very episode the column should show.

??? info "Details: the table box"
    - `nm_at_table_ratio` judges on each camera's table box: its work area as the last frame of the session holds it. It is learned offline, like the seats, so the start of a session, before the area is ready, is judged too.
    - A body is at the table when one of its hand circles is centred inside the box, or its box meets the box grown by one pupil shoulder width (about two hand lengths, an adult leaning in).
    - `nm_hands_in_table_ratio` counts only hands inside the box itself: someone outside the group handling the group's artifact. It is the more specific of the two, since a passer-by or a teacher watching also reaches the grown box.
    - A teacher the pose model missed, or one the rules take for the group, reads 0, not empty.

## The columns

The blocks below come in the table's order. Within a block, each person's columns come before each pair's.

### Window columns

| Column | What it holds |
|---|---|
| `window_index` | the window's number, from 0 |
| `window_start`, `window_end` | the window's start and end, in unix seconds; the keys other tables join on |

### Speech columns

| Column | What it holds |
|---|---|
| `n_asr_recognition` | the 3 s recognition buckets that overlap the window |
| `speech_ratio`, `silence_ratio` | the share of the window that held speech, and silence; a speaker several microphones heard in one bucket counts once, and no bucket counts for more than it covers |
| `n_speakers_named` | the named speakers heard |
| `spk_<name>_ratio` | the share of the window each named speaker spoke |
| `n_asr_transcription` | the transcript chunks that overlap the window |
| `n_spurts`, `mean_spurt_seconds` | the talk spurts (chunks) that started in the window, and their mean length |
| `words` | the words spoken in the window, each by its own stamp when the transcriber gave word timings, else the words of the chunks that started in it |
| `p<tag>_words` | worn microphones only: each wearer's words ([Word attribution](../pipelines/asr/speakers-and-diarization.md#word-attribution)) |
| `vote_words`, `p<tag>_vote_words` | worn microphones with level traces only: the bucket vote's word count summed (only without a group microphone), and per wearer |
| `dia_speakers`, `dia_switches`, `dia_overlap_ratio`, `dia_share_entropy` | the anonymous speaker turns of the diarized chunks: the most speakers of one reading, the switches and the overlapped time summed, the evenness of the shares averaged |

A session with worn microphones beside a group microphone reads `speech_ratio`, `silence_ratio`, `n_speakers_named`, `n_asr_transcription`, `words`, the spurts and `dia_*` from the group microphone alone. With worn microphones and no group microphone, `words` counts every word the worn microphones transcribed, each spoken word once; without level traces it is the sum of `p<tag>_words`. [Personal microphones](../pipelines/asr/speakers-and-diarization.md#personal-microphones-and-energy-attribution) gives the rules.

The diarized turns are read by session voice, across the window's chunks of one microphone and voice registry, when the base linked them ([Voices across chunks](../pipelines/asr/speakers-and-diarization.md#voices-across-chunks)); else per chunk, whose labels hold within it only. A speaker the base left without a voice counts per chunk too.

### Space columns

| Column | What it holds |
|---|---|
| `n_ips` | the IPS position records in the window |
| `p<tag>_present_ratio` | the share of those records that hold the tag |
| `p<tag>_path_m` | the metres the badge moved on the floor ([The path](#the-path)) |
| `pair<a>_<b>_dist_mean_m`, `_dist_min_m` | the mean and least distance between the two badges at the same moment, in metres |
| `n_ips_relation` | the IPS facing graphs in the window |
| `pair<a>_<b>_face_ab_ratio`, `_face_ba_ratio`, `_face_mutual_ratio` | of the graphs that hold both, the share in which a faced b, b faced a, and both |

### Body and gaze columns

| Column | What it holds |
|---|---|
| `n_vfa_features` | the frame sets in the window |
| `n_vfa_angles`, `n_vfa_cameras` | how many angle names and cameras the window's frames carry |
| `n_vfa_incomplete` | the frame sets whose number of frames differs from the one most of the session's sets have: a set that lost or gained a camera |
| `n_vfa_seat_partners` | the gaze frames, summed over the persons and cameras, whose untagged target counted as a partner by the seat rule ([Pupils](#pupils)); absent without VFA, or with `seat_partners=False` |
| `p<tag>_frames` | the person's camera frames; doubles with a second camera |
| `p<tag>_frame_sets`, `p<tag>_cameras` | the frame sets in which any camera saw the person, and the cameras that did |
| `p<tag>_yaw_mean`, `_yaw_abs_mean`, `_yaw_std` | the head yaw in degrees: its mean, its mean absolute value, and its spread |
| `p<tag>_wrist_speed` | the wrist speed, each hand followed from frame to frame, in frame widths per second |
| `p<tag>_gaze_switches` | the changes between the ten [gaze categories](#gaze-categories) |
| `p<tag>_gaze_<category>_ratio` | the share of the person's gaze frames in each of the ten [gaze categories](#gaze-categories); they sum to 1 |
| `p<tag>_work_area_ready_ratio` | the share of the person's gaze frames taken on a camera whose [work area](#the-work-area) was ready; 0 for events without keypoints |
| `p<tag>_in_group` | 1 when the tag is one of the session's [pupils](#pupils), else 0, on every row |
| `p<tag>_hand_steps` | the [steps](#the-hands-in-body-units) with a status |
| `p<tag>_hands_active_ratio`, `_hands_still_ratio` | the share of those steps that were active, and still |
| `p<tag>_wrist_speed_sw` | the mean wrist speed, in shoulder widths per second |
| `p<tag>_other_hands_near_ratio` | of the person's frames with a hand circle and a shoulder width, the share in which a hand circle of a body [outside the group](#who-is-outside-the-group) is centred within one hand length of one of theirs |
| `pair<a>_<b>_frames` | the camera frames that hold both |
| `pair<a>_<b>_frame_sets` | the frame sets in which one camera saw both together |
| `pair<a>_<b>_hand_dist_min`, `_hand_dist_mean` | the distance between the two persons' hands, in frame widths |
| `pair<a>_<b>_gaze_dist_mean` | the distance between the two gaze points, in frame widths |
| `pair<a>_<b>_joint_attention_ratio` | of the frames with a gaze distance for the pair, the share with the two gaze points within 5 % of the frame's width |
| `pair<a>_<b>_mutual_gaze_ratio` | the share of the frames in which each gaze is on the other's face |
| `pair<a>_<b>_joint_attention_baseline`, `_joint_attention_excess` | the pair's own rate of joint attention 20 to 40 s earlier, and the window's joint attention above it ([Joint attention against its own past](#joint-attention-against-its-own-past)) |
| `pair<a>_<b>_joint_member_ratio`, `_joint_outsider_ratio` | `-js` only: the share of the frames `_joint_attention_ratio` divides by in which the two gazes met on the group's own things, and on someone outside the group; the two sum to at most `_joint_attention_ratio` |
| `pair<a>_<b>_joint_member_baseline`, `_joint_member_excess` | `-js` only: the baseline counting only hits on the group's own things (at most `_joint_attention_baseline`), and the member ratio above it |
| `pair<a>_<b>_joint_reach_ratio`, `_joint_both_wa_ratio` | `-js` only, diagnostics over the same frames: outsider frames by the reach rule alone, and member frames with both gazes on the work area |
| `pair<a>_<b>_hand_steps` | the steps in which both had a status on the same camera |
| `pair<a>_<b>_one_active_ratio`, `_both_active_ratio`, `_both_still_ratio` | the share of those steps with one active and the other still, both active, and both still |
| `pair<a>_<b>_follow_ratio` | of the one-active steps, given at least three, the share in which the still pupil's gaze at the step's end is on the active one's hands |
| `pair<a>_<b>_hand_dist_sw_min`, `_hand_dist_sw_mean` | the hand distance over the mean of the two shoulder widths |
| `pair<a>_<b>_hands_close_ratio` | the share of those frames with the hands within one hand length |
| `n_vfa_propagated` | the person frames, summed over the cameras, whose tag the fusion carried [along their track](#tags-along-the-tracks); only in a session whose persons were tracked |

A ratio with nothing to judge is empty, never 0.

??? info "Details: how frames and cameras count"
    - Every frame of every camera that saw the person or the pair counts once in the gaze shares, the mean yaws and the pair values.
    - The wrist speed, the gaze switches and the yaw spread are taken within one camera, then averaged over the cameras (`_yaw_std` weighted by the frames each camera gave), so two cameras under one angle name do not read as movement.
    - A frame's camera is the `camera` the [features endpoint](../pipelines/vfa/pose-and-gaze.md#features-endpoint) names. A frame that names none is keyed by its angle, numbered by its place in the frame set when several cameras share the angle, so an incomplete set can give a frame to the wrong camera.
    - Mutual gaze reads the two persons' own targets: each `partner_face` naming the other.
    - `_follow_ratio` counts a gaze on the active one's hands as their `partner_hands` target, or a gaze point within reach of one of their hand circles, the gaze tolerance included, unless the server found the still pupil's own hands nearer. It says how much a still pupil's gaze was on a working partner's hands, not that the work drew it.
    - The `-js` columns are empty for a pair with someone outside the pupils, and the ratios are empty where `_joint_attention_ratio` is. `_joint_member_baseline` is empty below 10 comparisons.

### Gaze categories

| Category | The gaze lands on |
|---|---|
| `partner_face`, `partner_hands` | another [pupil's](#pupils) face or hands |
| `other_face`, `other_hands` | anyone else's face or hands: an untagged body (`track_<n>`, `unknown_<n>`) or a tag outside the pupils |
| `own_hands` | the person's own hands |
| `zone` | a configured zone |
| `work_area` | an `elsewhere` gaze inside the camera's [work area](#the-work-area) |
| `elsewhere` | anything else inside the frame |
| `out_of_frame` | outside the frame |
| `unknown` | nothing known |

The server's [gaze targets](../pipelines/vfa/pose-and-gaze.md#gaze-targets) say which target wins. A gaze moving from the work area to beyond it, or from a partner to the teacher, counts as a switch.

### Action columns

| Column | What it holds |
|---|---|
| `n_vfa_action` | the action labels in the window |
| `p<tag>_action` | the VLM's [action label](../pipelines/vfa/action-labels.md) for the person: the newest one up to the window's end, remembered for at most 60 s before its start |
| `pair<a>_<b>_co_manipulating` | 1 when both are labelled `Manipulating`, else 0; empty when either has no label |

### Seat and non-member columns

| Column | What it holds |
|---|---|
| `nm_at_table_ratio` | `-js` only: the share of the window's frame sets in which some camera saw a non-member at the group's table: a hand circle inside the camera's table box, or a box within one pupil shoulder width of it |
| `nm_hands_in_table_ratio` | `-js` only: the same, with a hand inside the table box; at most `nm_at_table_ratio` |
| `n_vfa_non_members` | `-js` only: the non-members summed over the window's frames, a diagnostic |
| `p<tag>_untagged_at_seat_ratio` | the share of the window's frame sets in which an untagged body stood at the person's [seat](#seats) while their tag was not seen on that camera |
| `n_untagged_at_seats` | the untagged bodies at any seat per frame set of the window, summed over the cameras |

The `nm_*` columns are empty when no camera of the window could judge (no table box learned, no pupil scale). In the seat trace, a frame set that lost the camera counts as one without such a body, and the cell is empty when no camera with the seat gave a frame. A session without VFA has none of these columns.

### Content columns

`mmla ses-fuse` does not write these. They come from a transcript reader: a local language model reads each window's transcript, with the 10 s before it as context, and a separate script appends its scores to a copy of the table, one score row per window.

| Column | What it holds |
|---|---|
| `content_teacher` | the probability that the teacher is speaking in the window |
| `content_peer_task` | the probability that peers talk about the task |
| `content_peer_other` | the probability that peers talk about something else |
| `content_no_text` | 1 when the window had no text, else 0; empty when the window has no score row |

A window with no text is not read, so its three probabilities are empty and count as 0. A table without these columns reads as one whose content was never scored.

## Analysis record

Each run writes `parameters.json` into a `fusion/` folder beside the table: `artifacts/<session>/analysis/features/fusion/parameters.json` by default. It names the software, the input files (with their sizes and digests) and the table, and its `parameters` hold:

| Key | What it holds |
|---|---|
| `window`, `step` | the window length and step |
| `participants` | the `-tags`, or null |
| `pupils`, `pupils_source` | the pupils, and `tags`, `manifest` or `trust bound` |
| `work_area` | the percentiles, the hand sightings before ready, and the padding |
| `seat_partners` | the seat rule for untagged gaze targets |
| `joint_baseline` | the `lags`, the `slack` and the least comparisons (`min`) |
| `hand_circle` | `version`, `relabelled`, `nudge_radii`, `radius_shoulders`, `stored_frames` (the frames by the circle the server made them with), and the stand-in thresholds |
| `hands` | the step bounds, the still and active bounds, the hand length, the follow minimum and the group tests |
| `speech` | which microphones the speech columns read and how worn words were counted ([Word attribution](../pipelines/asr/speakers-and-diarization.md#word-attribution)) |
| `joint_split` | `on`, and with it the member labels, the outsider and reach rules, and `non_members` (`scale` 0.6, `table_reach_sw` 1.0, the table box, the track gap, the seat rule) |
| `tag_memory_seconds` | the tag memory; null for no limit |
| `face_refusal_frames` | the face refusal run; null when off |
| `path_rule` | `on`, the plane, the floor's pitch and the constants; or the raw sum |
| `events` | the records read, per event type |
| `source` | the measurements folder, or `influxdb` |
| `cameras`, `frame_sets`, `window_span` | filtered runs only: what `-cams` and `-fs` kept, and the span the windows cover |

## Troubleshooting

**`No events found for session <id>: nothing to build a table from.`** The session id is wrong, the config's `InfluxDB` section points at another database, or the export folder holds no file of that session. Check the id on the **Sessions** tab.

**`give -c <config with an InfluxDB section> and -sid <session>, or --measurements <folder>`.** Reading from InfluxDB needs both `-c` and `-sid`.

**`could not tell the session from <folder>: give -sid <session>`.** Neither the file names nor the folder's name say the session. Give `-sid`.

**The partner shares are near 0 and `other_*` is high.** The pupils are wrong. Read the printed `Pupils (<source>)` line, and declare them with `mmla ses-tidy --pupils` or `-tags`.

**`<session>/manifest.json: pupils are tag ids, not ...`.** An entry of `pupils` in the session's manifest is not a tag id. Set them again with `mmla ses-tidy --pupils`.

**A pupil is absent in many windows.** Their tag is seen mostly through the server's track memory, which the [tag memory](#tags-along-the-tracks) cuts after 60 s. Check `n_vfa_propagated` and `p<tag>_untagged_at_seat_ratio`; a longer `-tm` keeps more of those frames, and wrong ones with them.

**The hand step columns are empty.** The frame sets are not 1 s apart ([The hands in body units](#the-hands-in-body-units)).

**`work_area` stays near 0 on one camera.** That camera sees few pupil hands, so its area is rarely ready; `p<tag>_work_area_ready_ratio` is low there.

**`-cams` or `-fs` refuses to run.** A filtered table needs `-o` outside `artifacts/<session>/analysis/features/`, every camera named must have frames in the session, and `-fs` takes two whole numbers with 1 ≤ KEEP < PERIOD.

**`Could not write the analysis record next to the table`.** The table is written and stands without its record. A filtered run (`-cams`, `-fs`) exits with an error instead.
