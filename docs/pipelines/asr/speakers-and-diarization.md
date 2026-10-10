# Speakers and diarization

This page covers whom ASR attributes speech to, and in what language it transcribes it: speaker profiles, anonymous speaker turns and their voices, and personal microphones. Read it when you set a base's **Participant**, **Speakers**, **Language** or **Diarize**.

## Attribution per base

Each base attributes its speech in one of three ways. Its base type's `asr_scope` sets what the base opens on, and the **Participant** you pick under the base on the ASR Base card's **Launch** tab decides for the session.

| `asr_scope` | **Participant** opens on | Whose the speech is |
|---|---|---|
| `individual` (the default) | **Speakers (speaker verification)** | the registered speaker each segment is recognized as, among the base's [speakers](#speakers) |
| `wearer` | the group's participants, in row order | the wearer's, every speech segment, without verification ([Personal microphones](#personal-microphones-and-energy-attribution)) |
| `group` | **Group** | the group's, without verification; [diarization](#diarize) tells the voices apart |

**Participant** offers the participants of the session's experiment group (`<name> (tag <id>)`), **Group** and **Speakers (speaker verification)**, and starts the base with `--participant <tag>|group|speakers`. The pick wins over the config for every run. A base with a wearer, whatever its `asr_scope`, verifies no speaker, labels every speech segment with the wearer's tag, and caps its chunks as a group microphone does ([Chunk length](#chunk-length)).

A row opens on the first of these that exists:

1. The pick kept for that experiment group and `Bases` entry. A pick made by hand holds for every session of the group.
2. The wearer the session's Collection Start noted for the base's stream (`wearers` in the session's MongoDB document).
3. What the base type's `asr_scope` says, as in the table.

??? info "Details: a base started without `--participant`"
    A base started from a terminal without `-pt`, or by the batch replay script, goes by the config:

    - the `participant` of its `Bases` entry (the Config tab does not show or keep this key);
    - else, for a stream, the wearer the session's Collection Start noted;
    - else its `asr_scope`.

    A `wearer` base that ends up with no wearer says so in yellow when it starts, and attributes its speech to the group for that run. The session's provenance notes where the wearer came from ([What a session records](configuration.md#what-a-session-records)).

## Speakers

A base whose **Participant** is **Speakers** recognizes who speaks among the speaker profiles registered on its host. The profiles live in one folder per speaker under `artifacts/runtime/pipelines/asr-base/<host>/profiles/`, shared by every base on that host until they are deleted. Which of them each base takes is its own: a **Speakers** line under the base says whom it takes at the next Start, and why.

By default a base takes the participants of the session's experiment group (Study → Experiments) that have a profile on the host. A profile stands for a participant when it is named as the participant or as their tag id, in any case. The group is the card's **Experiment Group** for `Create MongoDB Session`, else the group the picked session belongs to.

When no participant has a profile, the base takes every profile. A base in `capture` mode takes nobody.

**Manage** under such a base opens the profiles of the card's host:

| Control | What it does |
|---|---|
| a tick | a speaker the base takes; ticking one yourself makes the pick yours, kept for that `Bases` entry on that host, so a badge keeps its speakers whichever row it is on |
| **Use Group** | gives the pick back to the experiment group |
| **Use All**, **Use None** | ticks every profile, or none |
| **Delete** | removes the highlighted profile from the host, for every base there, on a second press |

### Register a speaker

Registration needs the ASR Server.

1. **Open Manage** under a Speakers base on the ASR Base card's **Launch** tab.
2. **Type the name.** The group's participants are suggested. A name that has a profile already adds to it.
3. **Record** the speaker from the source of the base picked under **Record from**, for **Seconds** (empty: the base type's `register_duration`), while they read the sentences shown. A stream the base pulls must be running: start it on the **Streams** tab first.
4. **Or Register Files**: add reference audio on this machine with **Add File…**. Files are copied to the card's host first when that is another machine.

Both go through VAD and noise reduction when the card has them on, and keep their audio in the profile only when **Store Audio** is on. A new profile is one more for every base of the host to tick. `mmla asr-speakers` does the same from a terminal ([Speaker profiles from the command line](run.md#speaker-profiles-from-the-command-line)).

## Language

**Language** on the Launch tab (`-lang`) is the language the bases of the card have their speech transcribed in. It travels with every transcription request and holds for that request alone, so changing it means starting the bases again; the speech transcriber keeps running, whatever its own config says. The first option, `the server's own language`, sends none, and the service transcribes in its configured language.

- The dropdown lists the common languages; `-lang` takes any code the backend knows (`yue`) or a locale (`en-GB`).
- Each backend is asked in its own form: a code for WhisperX, Whisper and DashScope, a locale for Azure (`da` becomes `da-DK`).
- The answer says which language was used, and a base says so in its window when the service did not take the one it asked for ([Troubleshooting](run.md#troubleshooting)).
- With `word_level` on, WhisperX aligns the words with a model of that language. The first use of a language fetches the model on the server, once; a language it has none for gives text without word timestamps.

## Diarization { #diarize }

**Diarize** on the Launch tab (`-dia`) sends every chunk for its anonymous speaker turns as well. The speech transcriber runs pyannote's diarization on the chunk, through WhisperX, so only a local `whisperx/` model can. The transcript record (`asr_transcription`) then carries `diarization`, the turns `[{start, end, speaker}]` in seconds from the start of the chunk, with the speakers named `SPEAKER_00`, `SPEAKER_01` ... within that chunk.

No profile, no name and no enrolment are needed. From the turns, a session's speaker changes, overlaps, active speakers per window and the evenness of their shares can be computed without knowing who anyone is; only which person a turn belongs to needs a profile, or another modality. Diarizing adds little to a chunk's transcription time.

| **Diarize** | `-dia` | Which bases diarize |
|---|---|---|
| `on for group microphones` (the default) | not passed | each base whose speech goes to the group: Participant **Group**, `asr_scope: group`, or a worn microphone nobody is noted as wearing |
| `on` | `true` | every base of the card |
| `off` | `false` | none |

A base decides at each chunk, so a worn microphone whose wearer the session notes stops diarizing for that run. To diarize every chunk of every base whatever it asks, set `SpeechTranscriber.local.diarize` on the server ([Transcription keys](configuration.md#transcription-keys)).

### Chunk length

A chunk is the audio of one speaker between two changes of speaker. A group or worn microphone never hears a change of speaker, so its chunk would end only at silence, which can be minutes away. `max_chunk_duration` in the base type's block caps it, at 30 s for `group` and `wearer` bases unless set ([Base blocks](configuration.md#base-blocks)). A chunk that reaches the cap is transcribed and diarized on its own, so the turns of a group microphone come per chunk of at most that length.

??? info "Details: where a capped chunk is cut"
    - A base without speaker verification (`group`, `wearer`) cuts a capped chunk at its quietest moment in the last 10 s before the cap: the middle of the 100 ms of lowest level, found every 10 ms in the chunk's raw audio (`openmmla/bases/asr/chunking.py`). The part before is transcribed, and the part after starts the next chunk.
    - The start time of that rest is counted back from the stamp of the segment that ended the chunk, so audio a live stream lost earlier in the chunk shifts no later chunk.
    - When the quietest stretch is the chunk's last 100 ms, the chunk goes whole. A chunk that ends at a silent segment is unchanged.
    - The cut reads only audio the base already has, so it waits for nothing and falls in the same place live and in replay. The words after the cut reach the database with the next chunk, up to one cap later.
    - An individual base, one that separates speech (`-sp`), and one whose cap is shorter than two segments (`recognize_duration`) cut at the cap and start the next chunk with the next segment. A cut never falls before half the cap, so only a cap of two segments or more keeps every rest shorter than the cap.

### Set up diarization

The pyannote pipeline is gated on huggingface.co. Accept its terms with an account, and give that account's token to the speech transcriber in one of two ways:

- as `SpeechTranscriber.local.hf_token` on the ASR Server card's **Config** tab, which stores it encrypted;
- as `HF_TOKEN` in `docker/.env` on the ASR Server's host, which `docker-compose.asr.yml` passes into the container. An exported variable does not reach a stack the console starts over SSH.

`diarize_model` picks the pipeline, and `min_speakers` and `max_speakers` bound the count when it is known ([Transcription keys](configuration.md#transcription-keys)).

## Voices across chunks

Each chunk is diarized on its own, so the `SPEAKER_00` of one chunk need not be the `SPEAKER_00` of the next. The base therefore links each chunk's speakers into the voices of its session, numbered 1, 2, 3 ... in the order they were first heard, per base and session.

- Every diarized word and turn of the transcript record carries its `voice` beside its `speaker`.
- The record carries `voices`, `{SPEAKER_NN: {voice, similarity}}`, with the similarity null for a voice the chunk started.
- Every linked record names the registry its voices come from (`voice_registry`: the base and the moment the registry began).
- Every linked record names the kind of speaker embedding its speakers were linked by (`voice_embedding`).

The speech transcriber returns, with the turns, `speaker_embeddings`: for each speaker of the chunk, an embedding of that speaker's own speech in it (256 numbers from the diarization pipeline's WeSpeaker ResNet34). It names their kind as `speaker_embedding_kind: speech`, and `/transcribe/info` names it too. A service that returns no embeddings writes records without `voices`; [Window features](../../analytics/window_features.md) then reads the turns per chunk.

!!! note
    A voice is a stable sound, not an identified person. A group microphone hears the group through the room, and one voice on it cannot be taken for one person. In a busy classroom it hears other groups and the teacher about as loudly as its own group, so its voices, and the `dia_*` features read from them, describe how much the room talks rather than who in the group talks. Per-person speech needs worn microphones ([Personal microphones](#personal-microphones-and-energy-attribution)).

??? info "Details: how speakers are linked into voices"
    - A speaker's speech is its turns minus every moment another speaker talks too, put one after the other and embedded whole. A speaker with less than 0.5 s of speech to itself is embedded from all of its turns, and one with less than 0.5 s in all comes without an embedding.
    - Pyannote's own centroid of each speaker is not used. Pyannote embeds 10 s windows and pads a shorter chunk with zeros, and the embedding model subtracts the mean of its features over the whole window, padding included, so one voice's centroids from chunks under 10 s and from longer ones hardly resemble each other.
    - A base compares embeddings of one kind only. A speech transcriber that names no kind returns pyannote's centroids, which the records name `voice_embedding: centroid`. The voices of a session take the kind and the length of its first voice's embeddings. When the kind changes in the middle of a session, because the speech transcriber was rebuilt, the base numbers the voices anew from there under a new `voice_registry`. A chunk of another length (another embedding model) is linked to none of them and starts none.
    - The base keeps the voices of its session (`openmmla/bases/asr/voices.py`). Each chunk's speakers are compared with the voices heard so far by the cosine similarity of their embedding to each voice's centroid, which is the voice's speakers so far weighted by their seconds of speech.
    - The most similar pairs are taken first, one speaker to one voice, since two speakers of one chunk are two voices. A pair counts at a similarity of 0.30 or above.
    - A speaker left over starts a new voice when it spoke at least 2 s in the chunk with nobody else talking, and has no voice otherwise. A speaker heard only over another is embedded from a mix of the two, so it may join a voice but never starts one.
    - The chunks are linked one at a time in the order they are transcribed, so a voice uses only the chunks before it, live and in replay alike.
    - A run restarted after a recording error keeps the voices of its session. A base launched again into the session cannot know them and numbers anew, under a new `voice_registry`.

## Personal microphones and energy attribution

A group can wear one microphone each, such as a badge channel per person, beside a room microphone, without anyone enrolling a voice. On synchronized channels worn by one person each, the wearer is the loudest voice on their own channel, and the neighbours reach it only as cross-talk, so the level of each channel tells them apart.

1. **Base type.** Give the worn microphones' kind `asr_scope: wearer` in its `Base` block ([Base blocks](configuration.md#base-blocks)).
2. **Wearers.** On the Launch tab, pick each worn microphone's wearer under **Participant** ([Attribution per base](#attribution-per-base)).
3. **The vote.** Two `Synchronizer` keys set it: `energy_margin_db` (6) and `energy_tie_db` (3) ([Synchronizer](configuration.md#synchronizer)).
4. **Recordings.** Bind the wearers of collected recordings with `mmla ses-tidy` ([Binding tags](#binding-tags)).

### Speech gate

A base counts a segment as speech in one of two ways, set by `speech_gate` in its `Base` block:

| `speech_gate` | A segment is speech when |
|---|---|
| `absolute` (the default) | VAD keeps speech in it, and its level after gain, noise reduction and VAD is over `rms_threshold` and `rms_peak_threshold` |
| `relative` | VAD keeps speech in it, and its raw level, before gain, stands at least `speech_gate_snr_db` (6) over the base's own noise floor |

A worn microphone can sit tens of dB below a room microphone, and each device has its own level, so a fixed threshold passes all of its segments or none: give such a microphone the `relative` gate. With it, gain and the rms thresholds no longer decide speech, but still shape what is transcribed.

The base prints its gate when it starts, and a blank or placeholder `speech_gate` or `speech_gate_snr_db` keeps its default. The gate works for any base; it does not tell the wearer from a neighbour who is just as loud, which is the synchronizer's vote.

### Noise floor and the vote

Each base measures the raw level of every segment, before any gain, and keeps its own noise floor: the 10th percentile of its segment levels over the last 60 s.

For each bucket, the synchronizer takes the personal channels with speech and how far each stands over its floor (`rms_db - floor_db`). The loudest wins when it is at least `energy_margin_db` up, and the others within `energy_tie_db` of it speak too. The losers are made silent before the merge, so they never come back as a fallback. The group microphone never votes, and its speech passes as it is.

??? info "Details: the floor at the start of a run"
    - Digital silence is left out of the floor, and a gap longer than 60 s starts it again.
    - Until five segments are held, the floor is the lowest level of the segments before the one it measures. A segment is never measured against its own level.
    - The first segment of a run, or after such a gap, has no floor (`floor_db: null`). It neither votes nor passes the relative gate, and its speech is kept as it is, like a recognition without `energy`.

### What the events carry

| Where | Field | What it holds |
|---|---|---|
| every recognition a base publishes on `<sid>/asr` | `energy` | `rms_db`, `peak_db` and `floor_db` of the raw segment, in dBFS |
| a wearer's recognition | `participant`, `levels` | the tag; the raw segment's level every 100 ms in dBFS (`db`, rounded to 0.1 dB), the step (`hop`, 0.1 s) and the segment's `floor_db` |
| `asr_recognition` | `energies` | JSON `{tag: snr_db}` of the personal channels that voted |
| `asr_recognition` | `levels` | JSON `{tag: {start, hop, floor_db, db}}` of every personal channel in the bucket, speech or silence, from its segment's start |
| a wearer's `asr_transcription` | `participant`, `attribution: energy`, `levels` | the tag; the chunk's raw level every 100 ms from its start and the base's floor when the chunk ended |

A wearer's chunk levels are taken in the order the segments arrive, so a live run and a replay give the same. Its transcript is stored a few microseconds after its chunk end, so the chunks of several bases that end together stay apart.

### Word attribution

A 3 s bucket is too coarse for worn microphones: they hear each other and people without a microphone, and the loudest often leads the next by only a few dB. The fusion (`mmla ses-fuse`) therefore decides each word of a worn microphone's transcript on its own (`attribute_word` in `openmmla/bases/asr/attribution.py`), from the level of every worn microphone over the word's span, less that microphone's floor. The margin is a fixed 6 dB (`ENERGY_MARGIN_DB`); the synchronizer's `energy_margin_db` does not change it. The word is:

- the **wearer's** when their microphone stands 6 dB over every other worn microphone and over its own floor;
- **cross-talk** when another wearer's microphone leads in the same way, so that wearer said it and it counts on their microphone;
- **nobody's** when no microphone leads: someone without a microphone, wearers talking over each other, or a word too quiet to tell.

At most one microphone leads at a moment. When the same word is on two microphones and each leads over its own, slightly different span, only the one that led by more keeps it (`count_once`: two wearer words of different wearers that overlap by at least half the shorter one).

??? info "Details: the levels a word is read from"
    - A word's level on a microphone is the power mean of the 100 ms steps its span covers. A microphone that never passed its speech gate counts too.
    - The levels come from the bucket's `levels` where they exist, and from the microphone's own transcript's `levels` otherwise. A word that runs past the end of one 3 s segment is read on into the next.
    - A microphone with no level at that moment counts as at its floor, as does one below it.
    - A session whose buckets carry no `levels` keeps the bucket vote, since the transcripts' levels alone would put every other microphone at its floor between its own chunks. A base run without a synchronizer is decided from its transcripts' levels.

??? info "Details: how the fusion counts words"
    - `mmla ses-fuse` adds a `p<tag>_words` column per wearer. In a session whose worn transcripts carry `levels`, these are the words that are the wearer's, and the bucket vote's count stays beside them in `p<tag>_vote_words`. Otherwise they are the words whose time falls in a 3 s bucket that lists the tag (and its `energies`, when the bucket has them).
    - Words without stamps are spread evenly over their chunk. A word the aligner gave no end lasts 0.3 s, and one it could not place takes the span of the word before it.
    - `words`, the spurts and `dia_*` stay the group microphone's. Without one, the spurts and `dia_*` come from every chunk, and `words` counts every word said near the microphones, whoever said it, each spoken word once (`once_across` in `attribution.py`).
    - A spoken word is on each microphone at most once, so a word stands for at most one word of each other microphone: a wearer's word first, then the word whose microphone led by most. A word that overlaps such a word of another microphone by at least half the shorter one is that word. The vote's sum stays beside it in `vote_words`; a session without levels keeps the sum of the wearers' vote words in `words`.
    - The analysis record next to the table says how the worn words were counted: `speech` with `worn_words` (`per word` or `bucket vote`), `word_margin_db` and `levels_from` (`buckets` or `transcripts`).

??? info "Details: how the fusion counts speech"
    - In a session with a group microphone beside worn ones, `speech_ratio`, `silence_ratio` and `n_speakers_named` read only the group microphone's entries of each `asr_recognition` bucket, so they measure what a session with the group microphone alone measures. A session with worn microphones only, or without worn microphones, reads every microphone's speech.
    - A merged bucket does not say which base an entry came from, but a worn microphone names its speech with its wearer's tag and the group microphone with the group's id (`group_01`), so the group's entries are the ones not named after a wearer. A group microphone whose own chunks carry a wearer's name (speaker verification against profiles named after the tags) cannot be told apart, and its session reads every microphone's speech.
    - The synchronizer drops a silent entry whenever another microphone heard speech, so a bucket with a wearer's speech and nothing from the group is one the group microphone called silent, or did not report. It counts as silent for its 3 s only when the group microphone named speech within 30 s before it and within 30 s after it. Otherwise (a group base that stopped, started late or dropped out) it counts neither speech nor silence, like a missing bucket, and a window left with no bucket to read has no `speech_ratio`, `silence_ratio` or `n_speakers_named`.
    - `spk_<name>_ratio` reads one name's entries, so a wearer's column is still their own microphone's speech.
    - The analysis record says which: `speech` with `from` (`group microphone` or `every microphone`), the `wearers`, the `group_silence_reach`, and `group_apart: false` for a group microphone that could not be told apart.

### Binding tags

Every audio recording of a session's manifests has a `scope` (`personal` or `group`) and a `participant`. A live Collection recording notes the wearer picked under the Collection form's **Participant**, else its device's default scope. What it left unbound is bound afterwards with `mmla ses-tidy --scope`, `--participant` and `--participants-in-order` ([Microphone scope and wearers](../../tui/launcher/collection/session-tools.md#microphone-scope-and-wearers)).

## Troubleshooting

**A base says it asked for turns and got none.** It says so once in its window, and the speech transcriber's log says which of these it is: the backend cannot diarize (not a local `whisperx/` model), the pyannote pipeline could not be made (the token or the accepted terms), or the service's image is out of date ([Troubleshooting](run.md#troubleshooting)).

**One person's short chunks are one voice and their long chunks another.** The speech transcriber returns pyannote's centroids: the records say `voice_embedding: centroid`, and the base says so once in its window. Its image is out of date: pull the latest OpenMMLA on the GPU server, then **Stop** and **Start** the ASR Server card, which builds the image anew. When the embeddings change in the middle of a session, the base says so and numbers the voices anew from there, under a new `voice_registry`.

**A registration gives no profile.** The message says why: which service does not answer and how (the Gateway has no route to it, its route has no server that answers, or nothing answers at the address), or, when they all answer, that the recording held no speech.

**A worn microphone's speech goes to the group.** The base found no wearer: pick one under its **Participant**, or bind the recording's wearer with `mmla ses-tidy`.
