# Docs style guide

How the pages under `docs/` are written. The site should read like the docs of classic open-source projects such as FastAPI, Home Assistant or MkDocs Material: neat, but complete and readable. This file is a contributor guide and is not published (it is in `exclude_docs` in `mkdocs.yml`). The VFA guide under `docs/pipelines/vfa/` follows it and serves as the worked example.

## Rules

### Page shape

- Open every page with one to three sentences: what this is and when you use it.
- Then the task or the concept, then the reference tables, then **Troubleshooting**.
- Split a page that covers several jobs (running, concepts, reference) into one page per job, and give the section an overview page that links to them.

### Guides of several pages

- A guide of several pages is a folder whose overview is `index.md`, served at the URL the single page had (`docs/tui/index.md` at `/tui/`). The overview ends with **Pages in this guide**, a list of its pages with one line each.
- A pipeline guide (`docs/pipelines/<name>/`) has the same pages as the VFA one:
    - `index.md`: **What <name> produces** (a short table), **How it works**, **Components**, **What you need**, **Pages in this guide**.
    - `run.md`: a `!!! note "Before you start"` that links to the Quickstart's one-time setup, then **Once per deployment**, **Every session**, **Run from the command line**, **Replay recordings** (anchor `#post-time-processing`) and **Troubleshooting**.
    - one page per topic (`speakers-and-diarization.md`, `calibration.md`, `pose-and-gaze.md`), and `configuration.md` with the config keys.
- New file names are lowercase words joined by hyphens (`live-video-and-sound.md`). Older names with underscores (`system_services.md`, `coding_interface.md`) stay, so their URLs keep working.
- Images live under `docs/img/<guide>/` and are linked relatively: from `docs/pipelines/ips/index.md` that is `../../img/ips/badge.png`.
- A link to a guide names its overview file (`tui/index.md`), not the folder.
- A task page tells the steps; the console's panel pages under `docs/tui/` own the tables of every field and button. When a task needs a panel's controls, it links to that table rather than repeating it.

### Paragraphs and lists

- One idea per paragraph, at most about four sentences, sentences of normal length.
- Options, keys, fields and steps go in lists and tables, not in prose. A reference table has the columns `Key | Default | What it does`.
- Steps the reader follows in order are a numbered list; everything else is a bulleted list.

### What the reader must not miss

- Use admonitions for it: `!!! note`, `!!! tip`, `!!! warning`. Give them a short title when the first words would not say what they are about.
- Rare edge cases, failure modes and implementation details that an operator may still need go into a collapsible block, closed by default, or into **Troubleshooting**. No operational fact is lost; the main text stays short.

### What does not belong in the docs

- How the docs or the code got here: no "used to", "earlier versions", "since <date>", "before <date>", pilot runs or commit history. Describe what the code does now.
- Evaluation results and study data: no accuracy or precision numbers, sample sizes, session, participant or camera-hour counts, ablation tables, or derivations of how the defaults were chosen. A default may keep a one-sentence reason in plain words.
- Unpublished studies, their classifiers and their codebooks.

### Wording

- Present tense and plain words. Address the reader as "you" in tasks.
- UI names in **bold** (**Start**, **Config** tab); keys, files, commands and values in backticks (`keyframe_interval`, `config.yml`, `mmla vfa-sync`).
- Commands and config examples in fenced code blocks with a language (`bash`, `yaml`, `json`).
- Menu paths as `Launcher → Pipelines → VFA → VFA Base`.
- One name per thing, as the console shows it. System services are `Role (Product)` on first mention in a page, then the role: **Stream Server (MediaMTX)**, then the Stream Server; **Gateway (Nginx)**, then the Gateway; **MQTT (Mosquitto)**; **Dashboard (Flask)** and **Dashboard (Celery)**. The card is **Session Control** and its buttons **Send START** and **Send STOP**. A label the UI spells otherwise (the dashboard's **Stream server** chip) keeps its own spelling.
- Example machines are `uber-server` (the system services), `gpu-server` (the AI services), `base-01`, `base-02` (base stations) and `pi-01`, `pi-02` (capture devices). Placeholders in configs are `<uber-server>`. No real host names, IP addresses, home paths or personal names.

### Headings

- Headings are short noun phrases or imperatives. They never contain " / "; write "and" or pick one word.
- No two headings on a page are the same, so every anchor is unique. Repeat the subject when needed ("Tracking settings", not a second "Settings").
- An anchor other pages link to stays stable. When a heading must change, keep the old anchor with an attribute: `## Replay recordings { #post-time-processing }`.

### One fact, one place

- Every fact lives on exactly one page. Elsewhere, link to it, with the target's heading as the link text when that reads well.
- Link to the exact section (`pose-and-gaze.md#tracking`), not just the page.
- Check the code when a fact is unclear; when the docs and the code disagree, the code wins.

### Figures

- Keep a figure where it helps understanding, and add none that only decorates.
- Its alt text says what the figure shows, not what it is called: "VFA Base card on the Launch tab: the base counts, the Session, a Base row per base, Mode, Graphics".

## Page template

```markdown
# <Name of the thing>

<What this is, in one sentence.> <When you use it, in one sentence.>

## <The task or the concept>

1. **<Step>.** <What to do, with the UI names in bold.>
2. **<Step>.** <...>

!!! warning
    <What goes wrong if the reader misses this.>

## <Reference>

| Key | Default | What it does |
|---|---|---|
| `<key>` | `<default>` | <one line> |

## Troubleshooting

**<The symptom, as the reader sees it.>** <The cause, and what to do.>
```

## Example: a folded detail

The main text gives the rule; the collapsible block keeps the edge cases for the operator who needs them:

```markdown
By default only a `different` verdict splits a track, because the face often has nothing
to compare, and splitting every unconfirmed re-find cuts the tracks of people who never left.

??? info "Details: splits and ByteTrack's matching"
    - `split_on: unconfirmed` splits every re-find after the gap that the appearance does
      not confirm. `split_gap_seconds: 0` never splits.
    - The tracker counts frames, and `frame_seconds` turns `split_gap_seconds` into frames.
```

Title a collapsible block `Details: <topic>`, and keep it closed (`???`, not `???+`). Put it right after the step or paragraph it belongs to, with a topic that names the case (`Details: what Start refuses on the Launch tab`), not a generic "More".

## Example: a reference table

```markdown
| Key | Default | What it does |
|---|---|---|
| `result_expiry_time` | `30` | seconds a frame set waits for missing bases; then it is sent with the frames it has, or dropped with only one |
| `match_tolerance` | `0.5` | seconds within which frames of different bases count as one moment |
| `action_interval` | `30` | the least seconds between two action-label requests; `0` asks for every frame set |
```

One row per key, the default as the code has it, and one line on what the key does. A key not in the shipped template says so in its row.

## Check a change

Build the site strictly from the repository root; it must pass without a warning:

```bash
mkdocs build --strict
```
