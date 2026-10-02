/**
 * The sound of a replay: one microphone recording of an ended session, played at the replay clock by
 * the rules the camera tiles' recorded video follows (cameras.js FilePlayer): its position is the
 * clock minus the file's start, it plays at the replay speed, pauses with the replay, seeks after a
 * jump, and is silent outside the file. At its 4x limit it cannot catch up by playing faster, so a lag
 * makes it seek a little ahead. Above 4x it is silent and lets go of its file: pitch-preserved speech
 * is unintelligible there (and browsers mute it themselves), and the file's connection is one of the
 * MAX_MEDIA the camera tiles share (takesMedia()).
 *
 * Nothing loads before the viewer's first gesture on the page's controls (arm(), called from Play,
 * Pause, Replay from start, unmuting, a pick in the control, or the note a blocked sound shows), so
 * autoplay rules never block it. A hidden page lets go of the file (release()). Follow mode has no
 * sound: the microphones stream AAC, which WebRTC does not carry.
 */

import { FilePlayer, driftTolerance, recordingAt, nextRecording, inlineUrl, RECORDING_LOOKAHEAD } from './cameras.js';

// the fastest replay speed with sound
export const MAX_SOUND_SPEED = 4;

function finite(v) {
  return typeof v === 'number' && Number.isFinite(v);
}

/**
 * the microphones among the recordings route's files: one source per device (a recorder that was
 * restarted left several files), each {key, label, scope, participant, files (by start), seconds}.
 * A file without a start cannot follow the clock and is left out.
 */
export function soundSources(files) {
  const by = new Map();
  for (const f of files || []) {
    if (!f || f.modality !== 'audio' || !finite(f.start) || !f.url) continue;
    const key = ['device', 'host', 'scope', 'participant'].map((k) => (f[k] == null ? '' : String(f[k]))).join('|');
    let src = by.get(key);
    if (!src) {
      src = { key, label: f.label || f.device || 'Microphone', host: f.host || null, scope: f.scope || null, participant: f.participant ?? null, files: [], seconds: 0 };
      by.set(key, src);
    }
    src.files.push(f);
    src.seconds += finite(f.duration) && f.duration > 0 ? f.duration : 0;
  }
  const out = Array.from(by.values());
  for (const src of out) src.files.sort((a, b) => a.start - b.start);
  // a label two sources share names their hosts
  const count = new Map();
  for (const src of out) count.set(src.label, (count.get(src.label) || 0) + 1);
  for (const src of out) if (count.get(src.label) > 1 && src.host) src.label = `${src.label} on ${src.host}`;
  return out;
}

/** the source a replay plays unless the viewer picks another: the group mic (the longest when there are several), else the first; null without any. */
export function defaultSource(sources) {
  const list = sources || [];
  let best = null;
  for (const src of list) if (src.scope === 'group' && (!best || src.seconds > best.seconds)) best = src;
  return best || list[0] || null;
}

/**
 * One <audio> element kept on the replay clock. update({source, now, speed, running, muted}) runs on
 * every redraw and returns what the sound does now: {state, message} with state 'off' (no source),
 * 'idle' (not armed yet), 'fast' (above MAX_SOUND_SPEED, no file held), 'gap' (no file at this
 * moment), 'loading', 'error', 'blocked' (the browser's autoplay rules refused it), 'playing' or
 * 'paused'; 'released' after release() until the next update.
 */
export class ReplaySound {
  constructor({ audio } = {}) {
    this.audio = audio;
    this.player = null;
    this.armed = false;
    this.file = null;
    this.status = { state: 'off', message: null };
    // a file's length as the browser read it, by url (it ends a file shorter than its listing says)
    this.durations = new Map();
  }

  /**
   * call from a click: the element may play from now on (a play() inside a gesture unlocks it where a
   * browser asks for one per element); again after the browser refused to play it
   */
  arm() {
    if (!this.audio || (this.armed && !(this.player && this.player.blocked))) return;
    this.armed = true;
    try {
      const r = this.audio.play();
      if (r && typeof r.catch === 'function') r.catch(() => {});
      this.audio.pause();
    } catch {
      // nothing to unlock
    }
  }

  /**
   * whether the sound takes one of the page's MAX_MEDIA connections to the dashboard: it holds a
   * file, or it is armed with a source at a speed it plays (it loads one on its next update)
   */
  takesMedia(source, speed) {
    const holds = !!(this.audio && typeof this.audio.getAttribute === 'function' && this.audio.getAttribute('src'));
    return holds || (this.armed && !!source && !(finite(speed) && speed > MAX_SOUND_SPEED));
  }

  /** let go of the file (hidden page, Off, follow mode, above 4x); the next update loads it again */
  release() {
    if (this.player) {
      this.player.stop();
      this.player = null;
    }
    this.file = null;
    this.set('released');
  }

  update({ source = null, now = null, speed = 1, running = false, muted = false } = {}) {
    if (!source || !this.audio) {
      this.release();
      return this.set('off');
    }
    if (!this.armed) {
      this.release();
      return this.set('idle');
    }
    const s = finite(speed) && speed > 0 ? speed : 1;
    if (s > MAX_SOUND_SPEED) {
      // silent this fast: the file's connection goes back to the camera tiles
      this.release();
      return this.set('fast');
    }
    if (!this.player) {
      // the replay sets muted itself (the toggle); no nudge asks for more than 4x, so a lag at 4x
      // closes by a seek a little ahead (catchUp)
      this.player = new FilePlayer({ video: this.audio, muted: false, maxRate: MAX_SOUND_SPEED, catchUp: true });
      this.player.start();
    }
    const p = this.player;
    if (p.source && p.duration != null) this.durations.set(p.source, p.duration);
    const file = finite(now) ? recordingAt(source.files, now, this.durations) : null;
    const next = file || !finite(now) ? null : nextRecording(source.files, now, RECORDING_LOOKAHEAD * s);
    const pick = file || next;
    // a file loads from the clock's moment (a coming one from its start)
    if (pick) p.load(inlineUrl(pick.url), pick.url, pick === file ? now - file.start : 0);
    else if (p.source && !source.files.some((f) => f.url === p.source)) {
      // another source's file: let it go
      p.stop();
      p.start();
    }
    this.file = file;
    const target = file && p.source === file.url ? now - file.start : null;
    p.sync({ target, speed: s, playing: !!running, tolerance: driftTolerance(s) });
    this.audio.muted = !!muted;
    if (p.state === 'error') return this.set('error', p.message);
    if (!file) return this.set('gap');
    if (!p.hadFrame || p.state === 'loading') return this.set('loading');
    if (p.blocked && running) return this.set('blocked');
    return this.set(running && !this.audio.paused ? 'playing' : 'paused');
  }

  set(state, message = null) {
    this.status = { state, message };
    return this.status;
  }

  /** the element as it stands, for checks from the browser console */
  inspect() {
    const a = this.audio;
    const p = this.player;
    return {
      armed: this.armed, state: this.status.state, message: this.status.message, file: this.file ? this.file.id : null,
      src: a ? a.getAttribute('src') : null, target: p ? p.target : null, time: a ? a.currentTime : null,
      paused: a ? a.paused : null, muted: a ? a.muted : null, rate: a ? a.playbackRate : null, stepping: p ? p.stepping : false,
      seeks: p ? p.seeks : 0,
    };
  }
}
